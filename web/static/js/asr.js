/* 语音输入（本地识别）：录音后交给项目二的 SenseVoiceSmall 识别成文字。
 *
 * 和 voice.js 的区别
 * ------------------
 * voice.js 用的是浏览器内置的 Web Speech API：识别在厂商的在线服务里做，
 * 断网就不可用，而且内置浏览器里那个对象常常是个"空壳"。
 * 本文件改成：浏览器只负责采集音频，识别交给项目二的本地模型
 * （multimodal_ai_project1 的桥接服务 :8100），不联网、中文短句也更稳。
 *
 * 链路：
 *   getUserMedia（麦克风）→ AudioContext 采 PCM → 重采样到 16k 单声道
 *   → 前端编码成 WAV → POST /api/voice → 项目一转发给桥接服务 → 文本
 *
 * 为什么在前端编码 WAV：项目二那边只要 16k 单声道 WAV，而浏览器 MediaRecorder
 * 给的是 webm/opus，服务端还得装 ffmpeg 转码；在浏览器里直接算 PCM 更省事。
 *
 * 失败时不会让页面卡住：桥接服务没启动会给出 bridge-down 错误码，
 * 页面据此切回 voice.js 的浏览器识别（降级但不报错）。
 */
(function () {
  "use strict";

  var TARGET_RATE = 16000; // SenseVoiceSmall 要 16k
  var MAX_SECONDS = 15; // 一次录音的上限，避免一直录下去
  var MIN_SECONDS = 0.3; // 太短的录音直接忽略（误点）

  function noop() {}

  /* 把 Float32 PCM 编成 16bit 单声道 WAV（Blob）。 */
  function encodeWav(samples, sampleRate) {
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);
    function writeString(offset, text) {
      for (var i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
    }
    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true); // fmt chunk 长度
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // 单声道
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // 字节率
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // 位深
    writeString(36, "data");
    view.setUint32(40, samples.length * 2, true);
    var offset = 44;
    for (var i = 0; i < samples.length; i++, offset += 2) {
      var s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([view], { type: "audio/wav" });
  }

  /* 线性插值重采样：浏览器给 44.1k/48k 时降到 16k。 */
  function resample(input, fromRate, toRate) {
    if (fromRate === toRate || !input.length) return input;
    var ratio = fromRate / toRate;
    var length = Math.round(input.length / ratio);
    var output = new Float32Array(length);
    for (var i = 0; i < length; i++) {
      var pos = i * ratio;
      var left = Math.floor(pos);
      var right = Math.min(left + 1, input.length - 1);
      var weight = pos - left;
      output[i] = input[left] * (1 - weight) + input[right] * weight;
    }
    return output;
  }

  /* 合并分块 + 统计峰值音量（用来判断"有没有听到声音"）。 */
  function mergeChunks(chunks) {
    var total = 0;
    var i;
    for (i = 0; i < chunks.length; i++) total += chunks[i].length;
    var merged = new Float32Array(total);
    var offset = 0;
    var peak = 0;
    for (i = 0; i < chunks.length; i++) {
      merged.set(chunks[i], offset);
      for (var j = 0; j < chunks[i].length; j++) {
        var abs = Math.abs(chunks[i][j]);
        if (abs > peak) peak = abs;
      }
      offset += chunks[i].length;
    }
    return { samples: merged, peak: peak };
  }

  function LocalASR(options) {
    options = options || {};
    this.endpoint = options.endpoint || "/api/voice";
    this.maxSeconds = options.maxSeconds || MAX_SECONDS;
    this.onResult = options.onResult || noop;
    this.onStatus = options.onStatus || noop;
    this.onError = options.onError || noop;

    this.recording = false;
    this._chunks = [];
    this._ctx = null;
    this._stream = null;
    this._node = null;
    this._source = null;
    this._timer = null;
    this._startedAt = 0;
  }

  LocalASR.prototype.supported = function () {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && Ctx);
  };

  /* 开始录音。返回 Promise<boolean>。 */
  LocalASR.prototype.start = function () {
    var self = this;
    if (this.recording) return Promise.resolve(true);
    if (!this.supported()) {
      this.onStatus("unsupported", "当前浏览器不支持录音采集，请改用键盘输入");
      return Promise.resolve(false);
    }
    var precheck = window.DeviceUtil ? DeviceUtil.has("audioinput") : Promise.resolve(true);
    return precheck.then(function (exists) {
      if (!exists) {
        self.onStatus(
          "nodevice",
          "没有检测到麦克风设备。本地语音识别需要麦克风；" +
            "请插上麦克风后点「检测设备」，或直接改用键盘输入。"
        );
        self.onError("no-device");
        return false;
      }
      return self._openMic();
    });
  };

  LocalASR.prototype._openMic = function () {
    var self = this;
    return navigator.mediaDevices
      .getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      })
      .then(function (stream) {
        self._stream = stream;
        var Ctx = window.AudioContext || window.webkitAudioContext;
        var ctx;
        try {
          // 直接要 16k：浏览器支持的话后面就不必重采样
          ctx = new Ctx({ sampleRate: TARGET_RATE });
        } catch (e) {
          ctx = new Ctx();
        }
        self._ctx = ctx;
        self._source = ctx.createMediaStreamSource(stream);
        // ScriptProcessorNode 虽然被标记为过时，但兼容性最好、不需要额外的
        // AudioWorklet 文件；对"录一段话"这种低要求的场景足够。
        self._node = ctx.createScriptProcessor(4096, 1, 1);
        self._chunks = [];
        self._node.onaudioprocess = function (event) {
          if (!self.recording) return;
          var input = event.inputBuffer.getChannelData(0);
          self._chunks.push(new Float32Array(input)); // 必须复制：缓冲区会被复用
        };
        self._source.connect(self._node);
        self._node.connect(ctx.destination);
        self.recording = true;
        self._startedAt = Date.now();
        self.onStatus("listening", "正在录音…说完再点一次停止（最长 " + self.maxSeconds + " 秒）");
        // 到点自动停，避免用户忘了点
        self._timer = setTimeout(function () {
          if (self.recording) self.stop();
        }, self.maxSeconds * 1000);
        return true;
      })
      .catch(function (err) {
        self.onStatus(
          "error",
          window.DeviceUtil
            ? DeviceUtil.explainError(err, "麦克风")
            : "无法打开麦克风：" + (err && err.message ? err.message : "未知错误")
        );
        self.onError((err && err.name) || "unknown");
        return false;
      });
  };

  /* 停止录音并把音频送去识别。 */
  LocalASR.prototype.stop = function () {
    if (!this.recording) return;
    this.recording = false;
    clearTimeout(this._timer);
    this._timer = null;

    var ctx = this._ctx;
    var rate = (ctx && ctx.sampleRate) || TARGET_RATE;
    try {
      if (this._node) this._node.disconnect();
      if (this._source) this._source.disconnect();
    } catch (e) {
      /* 忽略 */
    }
    if (this._stream) {
      this._stream.getTracks().forEach(function (t) {
        t.stop();
      });
      this._stream = null;
    }
    if (ctx && ctx.close) {
      try {
        ctx.close();
      } catch (e) {
        /* 忽略 */
      }
    }
    this._ctx = null;
    this._node = null;
    this._source = null;

    var seconds = (Date.now() - this._startedAt) / 1000;
    var merged = mergeChunks(this._chunks);
    this._chunks = [];
    if (seconds < MIN_SECONDS || !merged.samples.length) {
      this.onStatus("error", "录音太短，请说完整一句再停");
      this.onError("too-short");
      return;
    }
    if (merged.peak < 0.01) {
      this.onStatus("error", "没有听到声音，请靠近麦克风再说一次");
      this.onError("silent");
      return;
    }

    var samples = resample(merged.samples, rate, TARGET_RATE);
    var blob = encodeWav(samples, TARGET_RATE);
    this._upload(blob);
  };

  LocalASR.prototype._upload = function (blob) {
    var self = this;
    this.onStatus("uploading", "正在用本地模型识别（项目二 SenseVoiceSmall）…");
    var started = Date.now();
    fetch(this.endpoint, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: blob,
    })
      .then(function (resp) {
        return resp
          .json()
          .catch(function () {
            return {};
          })
          .then(function (data) {
            return { ok: resp.ok, status: resp.status, data: data };
          });
      })
      .then(function (r) {
        if (!r.ok) {
          // 503 = 桥接服务没启动：交给页面回退到浏览器识别
          var fallback = r.status === 503 || r.data.fallback === "builtin";
          self.onStatus("error", r.data.error || "语音识别失败");
          self.onError(fallback ? "bridge-down" : "server");
          return;
        }
        var text = (r.data.text || "").trim();
        var cost = Date.now() - started;
        if (!text) {
          self.onStatus("error", "没有识别到语音内容，请靠近麦克风再说一次");
          self.onError("empty");
          return;
        }
        self.onResult(text, true);
        self.onStatus("done", "识别完成（本地模型，" + cost + " ms），点「用这句话搜索」即可查询");
      })
      .catch(function (err) {
        self.onStatus("error", "上传失败：" + err.message);
        self.onError("network");
      });
  };

  LocalASR.prototype.toggle = function () {
    if (this.recording) {
      this.stop();
      return Promise.resolve(false);
    }
    return this.start();
  };

  window.LocalASR = LocalASR;
})();
