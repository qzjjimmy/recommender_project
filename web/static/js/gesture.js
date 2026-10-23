/* 手势识别：用摄像头判断"竖大拇指(喜欢) / 拇指向下(不喜欢)"。
 *
 * 为什么不用 MediaPipe
 * -------------------
 * 本项目环境里 jsdelivr / unpkg 这类 CDN 被网络策略拦截，国内 CDN 也没有托管
 * MediaPipe 的 wasm 与模型文件，安装 Python 版又需要联网下载。为了让功能"打开就能用、
 * 不依赖任何外部资源"，这里用**纯 JavaScript + Canvas 像素分析**实现：
 *
 *     1. 把摄像头画面降采样到 160×120；
 *     2. 按 YCbCr 肤色范围做二值分割，得到"疑似手部"的掩码；
 *     3. 找最大的连通区域（去掉噪点与其它肤色区域）；
 *     4. 统计该区域**每一行的宽度**，得到"形状宽度轮廓"；
 *     5. 竖大拇指时：拳头在下（宽）、拇指在上（窄）→ 上窄下宽；
 *        拇指向下时：拳头在上（宽）、拇指在下（窄）→ 上宽下窄；
 *     6. 连续稳定识别 holdMs 毫秒才触发，避免误触。
 *
 * 这是一个**几何规则**方案（和指导书里"基于关键点 y 坐标关系判断手指状态"的思路一致），
 * 精度不如 MediaPipe，但对"竖大拇指/倒大拇指"这种形状差别很大的手势足够用。
 * 界面上会把掩码和宽度轮廓画出来，识别过程完全可见、可调试。
 */
(function () {
  "use strict";

  var PW = 160; // 处理宽度
  var PH = 120; // 处理高度
  var SHAPE_RATIO = 1.35; // 上/下宽度比超过这个倍数才判定

  function isSkin(r, g, b) {
    var y = 0.299 * r + 0.587 * g + 0.114 * b;
    if (y < 60) return false;
    var cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    var cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    return cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173;
  }

  function GestureRecognizer(options) {
    options = options || {};
    this.video = options.video;
    this.maskCanvas = document.createElement("canvas"); // 处理用，不显示
    this.maskCanvas.width = PW;
    this.maskCanvas.height = PH;
    this.overlay = options.overlay; // 显示用
    this.holdMs = options.holdMs || 900;
    this.minConfidence = options.minConfidence || 0.20;
    this.cooldownMs = options.cooldownMs || 1500;
    this.onGesture = options.onGesture || function () {};
    this.onStatus = options.onStatus || function () {};

    this.stream = null;
    this.running = false;
    this.stableLabel = null;
    this.stableSince = 0;
    this.lastFiredAt = 0;
    this.lastResult = null;
  }

  GestureRecognizer.prototype.supported = function () {
    return !!(window.navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  };

  GestureRecognizer.prototype.start = function () {
    var self = this;
    if (this.running) return Promise.resolve(true);
    if (!this.supported()) {
      this.onStatus("error", "当前浏览器不支持摄像头调用，请改用「喜欢 / 不喜欢」按钮");
      return Promise.resolve(false);
    }
    // 先判断设备存不存在：没有摄像头时直接说明，不要抛一个容易被误读的错误
    var precheck = window.DeviceUtil ? DeviceUtil.has("videoinput") : Promise.resolve(true);
    return precheck.then(function (exists) {
      if (!exists) {
        self.onStatus(
          "nodevice",
          "没有检测到摄像头设备。手势反馈需要摄像头；\n" +
        "可以改用每条结果右侧的「喜欢 / 不喜欢」按钮，或插上 USB 摄像头后点击「检测设备」。"
        );
        return false;
      }
      return self._openCamera();
    });
  };

  /* 逐级放宽约束打开摄像头：facingMode 在部分台式机上会直接失败，所以留了降级链。 */
  GestureRecognizer.prototype._openCamera = function () {
    var self = this;
    var attempts = [
      { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false },
      { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
      { video: true, audio: false },
    ];

    function tryAt(index, lastError) {
      if (index >= attempts.length) {
        self.onStatus("error", self._explain(lastError));
        return false;
      }
      return navigator.mediaDevices
        .getUserMedia(attempts[index])
        .then(function (stream) {
          self.stream = stream;
          self.video.srcObject = stream;
          return self.video.play();
        })
        .then(function () {
          self.running = true;
          var label = "";
          try {
            var track = self.stream.getVideoTracks()[0];
            label = track && track.label ? "（" + track.label + "）" : "";
          } catch (e) {
            /* 忽略 */
          }
          self.onStatus("running", "摄像头已开启" + label + "，把手举到画面中央，拇指朝上或朝下");
          self._loop();
          return true;
        })
        .catch(function (err) {
          var name = (err && err.name) || "";
          // 只有"参数不合适/设备暂时用不了"才换更宽松的参数重试；
          // 权限被拒（NotAllowedError）不该反复弹窗
          if (name === "OverconstrainedError" || name === "NotFoundError" || name === "NotReadableError") {
            return tryAt(index + 1, err);
          }
          self.onStatus("error", self._explain(err));
          return false;
        });
    }
    return tryAt(0, null);
  };

  GestureRecognizer.prototype._explain = function (err) {
    return window.DeviceUtil
      ? DeviceUtil.explainError(err, "摄像头")
      : "无法打开摄像头：" + (err && err.message ? err.message : "未知错误");
  };

  GestureRecognizer.prototype.stop = function () {
    this.running = false;
    if (this.stream) {
      this.stream.getTracks().forEach(function (t) {
        t.stop();
      });
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;
    if (this.overlay) {
      var ctx = this.overlay.getContext("2d");
      ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    }
    this.stableLabel = null;
    this.onStatus("stopped", "摄像头已关闭");
  };

  GestureRecognizer.prototype.toggle = function () {
    if (this.running) {
      this.stop();
      return Promise.resolve(false);
    }
    return this.start();
  };

  // ------------------------------------------------------------ 主循环
  GestureRecognizer.prototype._loop = function () {
    var self = this;
    if (!this.running) return;
    this._tick();
    // 约 12 fps，够用且省 CPU
    this._timer = setTimeout(function () {
      self._loop();
    }, 80);
  };

  GestureRecognizer.prototype._tick = function () {
    var v = this.video;
    if (!v || v.readyState < 2) return;

    var ctx = this.maskCanvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(v, 0, 0, PW, PH);
    var frame = ctx.getImageData(0, 0, PW, PH);
    var data = frame.data;

    var mask = new Uint8Array(PW * PH);
    var maskImage = ctx.createImageData(PW, PH);
    var md = maskImage.data;
    var skinCount = 0;

    for (var i = 0, p = 0; i < mask.length; i++, p += 4) {
      var r = data[p],
        g = data[p + 1],
        b = data[p + 2];
      if (isSkin(r, g, b)) {
        mask[i] = 1;
        skinCount++;
        // 掩码用半透明绿色显示，方便用户看到"系统认为哪里是皮肤"
        md[p] = 40;
        md[p + 1] = 220;
        md[p + 2] = 120;
        md[p + 3] = 140;
      }
    }

    var blob = this._largestBlob(mask);
    var result = { label: null, confidence: 0, blob: blob, profile: null, skinCount: skinCount };
    if (blob && blob.area >= 250 && blob.height >= PH * 0.35) {
      var shape = this._analyzeProfile(blob, mask);
      if (shape) {
        result.profile = shape.profile;
        result.confidence = shape.confidence;
        result.label = shape.label;
      }
    }

    this.lastResult = result;
    this._draw(result, blob, maskImage);
    this._decide(result);
  };

  // -------------------------------------------------- 找最大连通区域
  GestureRecognizer.prototype._largestBlob = function (mask) {
    var visited = new Uint8Array(PW * PH);
    var best = null;
    var stack = new Int32Array(PW * PH);

    for (var start = 0; start < mask.length; start++) {
      if (!mask[start] || visited[start]) continue;

      var top = 0;
      stack[top++] = start;
      visited[start] = 1;

      var area = 0,
        x0 = PW,
        y0 = PH,
        x1 = -1,
        y1 = -1;
      var rows = new Int32Array(PH);
      var pixels = [];

      while (top > 0) {
        var idx = stack[--top];
        var x = idx % PW,
          y = (idx - x) / PW;
        area++;
        rows[y]++;
        pixels.push(idx);
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;

        if (x > 0 && mask[idx - 1] && !visited[idx - 1]) {
          visited[idx - 1] = 1;
          stack[top++] = idx - 1;
        }
        if (x < PW - 1 && mask[idx + 1] && !visited[idx + 1]) {
          visited[idx + 1] = 1;
          stack[top++] = idx + 1;
        }
        if (y > 0 && mask[idx - PW] && !visited[idx - PW]) {
          visited[idx - PW] = 1;
          stack[top++] = idx - PW;
        }
        if (y < PH - 1 && mask[idx + PW] && !visited[idx + PW]) {
          visited[idx + PW] = 1;
          stack[top++] = idx + PW;
        }
      }

      if (!best || area > best.area) {
        best = {
          area: area,
          x0: x0,
          y0: y0,
          x1: x1,
          y1: y1,
          width: x1 - x0 + 1,
          height: y1 - y0 + 1,
          rows: rows,
        };
      }
    }
    return best;
  };

  // -------------------------------------------- 形状宽度轮廓 → 手势判定
  GestureRecognizer.prototype._analyzeProfile = function (blob, mask) {
    var h = blob.height;
    if (h < 20) return null;

    // 取每一行的宽度（只看该连通区域的行计数）
    var profile = [];
    for (var y = blob.y0; y <= blob.y1; y++) profile.push(blob.rows[y]);

    // 轻度平滑，抵消边缘抖动
    var smooth = profile.map(function (_, i) {
      var a = profile[Math.max(0, i - 1)],
        b = profile[i],
        c = profile[Math.min(profile.length - 1, i + 1)];
      return (a + b + c) / 3;
    });

    var band = Math.max(2, Math.round(smooth.length * 0.25));
    var mean = function (arr) {
      var s = 0;
      for (var i = 0; i < arr.length; i++) s += arr[i];
      return s / arr.length;
    };
    var wTop = mean(smooth.slice(0, band));
    var wBottom = mean(smooth.slice(smooth.length - band));
    if (wTop <= 0 || wBottom <= 0) return null;

    var ratio = wBottom / wTop;
    var confidence = Math.abs(wBottom - wTop) / Math.max(wBottom, wTop);

    var label = null;
    if (ratio >= SHAPE_RATIO) label = "like"; // 上窄下宽 = 拳头在下、拇指朝上
    else if (ratio <= 1 / SHAPE_RATIO) label = "dislike"; // 上宽下窄 = 拇指朝下

    return {
      label: label,
      confidence: confidence,
      ratio: ratio,
      wTop: wTop,
      wBottom: wBottom,
      profile: smooth,
      box: blob,
    };
  };

  // ------------------------------------------------------------ 触发判定
  GestureRecognizer.prototype._decide = function (result) {
    var now = Date.now();
    var label = result.label;
    var confident = result.confidence >= this.minConfidence;

    if (!label || !confident) {
      this.stableLabel = null;
      this._progress = 0;
      if (result.blob) {
        this.onStatus(
          "tracking",
          label ? "手势不够明确，请让拇指更明显地朝上或朝下" : "已检测到手，请竖起大拇指或拇指向下"
        );
      } else {
        this.onStatus("searching", "把手举到画面中央，与背景拉开对比");
      }
      return;
    }

    if (this.stableLabel !== label) {
      this.stableLabel = label;
      this.stableSince = now;
      this._progress = 0;
      return;
    }

    var held = now - this.stableSince;
    this._progress = Math.min(1, held / this.holdMs);
    if (held < this.holdMs) {
      this.onStatus(
        "holding",
        (label === "like" ? "识别到「喜欢」" : "识别到「不喜欢」") +
          "，保持 " + ((this.holdMs - held) / 1000).toFixed(1) + " 秒确认"
      );
      return;
    }
    if (now - this.lastFiredAt < this.cooldownMs) return;

    this.lastFiredAt = now;
    this.stableLabel = null;
    this._progress = 0;
    this.onGesture(label, result);
  };

  // ------------------------------------------------------------ 画面绘制
  GestureRecognizer.prototype._draw = function (result, blob, maskImage) {
    var ov = this.overlay;
    if (!ov) return;
    var octx = ov.getContext("2d");
    var sx = ov.width / PW,
      sy = ov.height / PH;

    octx.clearRect(0, 0, ov.width, ov.height);
    // 1) 掩码：让用户看到系统识别到的"手"
    octx.save();
    octx.globalAlpha = 0.45;
    octx.drawImage(this.maskCanvas, 0, 0, ov.width, ov.height);
    octx.restore();

    if (!blob) return;

    // 2) 外接框
    octx.strokeStyle = result.label ? "#22c55e" : "#f59e0b";
    octx.lineWidth = 2;
    octx.strokeRect(blob.x0 * sx, blob.y0 * sy, blob.width * sx, blob.height * sy);

    // 3) 宽度轮廓：把"算法看到的手形"画出来，识别过程完全透明
    if (result.profile) {
      var maxW = 0;
      for (var i = 0; i < result.profile.length; i++) {
        if (result.profile[i] > maxW) maxW = result.profile[i];
      }
      if (maxW > 0) {
        octx.strokeStyle = "rgba(59,130,246,0.95)";
        octx.lineWidth = 1;
        octx.beginPath();
        for (var k = 0; k < result.profile.length; k++) {
          var y = (blob.y0 + k) * sy;
          var w = (result.profile[k] / maxW) * blob.width * sx;
          octx.moveTo(blob.x0 * sx, y);
          octx.lineTo(blob.x0 * sx + w, y);
        }
        octx.stroke();
      }
    }

    // 4) 判定文字
    octx.font = "bold 15px sans-serif";
    octx.fillStyle = result.label === "like" ? "#22c55e" : result.label === "dislike" ? "#ef4444" : "#f59e0b";
      var text = result.label === "like" ? "喜欢" : result.label === "dislike" ? "不喜欢" : "识别中";
    octx.fillText(text, 8, 20);
    octx.font = "12px sans-serif";
    octx.fillStyle = "#e5e7eb";
    octx.fillText("置信度 " + (result.confidence * 100).toFixed(0) + "%", 8, 38);
  };

  GestureRecognizer.prototype.progress = function () {
    return this._progress || 0;
  };

  /* ==================================================================
   * GestureBridge：把摄像头画面送去项目二的 MediaPipe 做手势判定
   * ------------------------------------------------------------------
   * 和上面那套纯 JS 算法的区别：那套用"肤色分割 + 宽度轮廓"猜形状，精度有限；
   * 这套把每一帧 JPEG 发给项目二的桥接服务（/api/gesture），由 MediaPipe
   * 的 21 个手部关键点判定，精度高得多。
   *
   * 分工：
   *   后端只做"单帧判定"（返回 thumbs_up / thumbs_down / null）；
   *   前端负责"稳定保持 + 冷却" —— 连续的判定要持续 holdMs 才触发一次，
   *   触发后 cooldownMs 内不再响应（跟原来那套的手感保持一致）。
   *
   * 对外接口与 GestureRecognizer 完全一致（running / toggle / stop /
   * onStatus / onGesture），页面上可以按桥接服务的可用性二选一。
   * ================================================================== */
  function GestureBridge(options) {
    GestureRecognizer.call(this, options);
    options = options || {};
    this.endpoint = options.endpoint || "/api/gesture";
    this.fps = options.fps || 5;
    // 桥接服务中途挂掉时回调（页面据此切回内置算法）
    this.onFallback = options.onFallback || null;
    // 送帧用的画布：小一点省带宽也够 MediaPipe 用（关键点是归一化坐标）
    this.frameCanvas = document.createElement("canvas");
    this.frameCanvas.width = 320;
    this.frameCanvas.height = 240;
    this._sending = false;
    this._winLabel = null;
    this._winSince = 0;
    this._lastReply = null;
    this._lastError = "";
    this.bridgeDown = false;
  }

  GestureBridge.prototype = Object.create(GestureRecognizer.prototype);
  GestureBridge.prototype.constructor = GestureBridge;

  /* 比本地算法慢一些：MediaPipe 单帧要几十毫秒，再快也没意义 */
  GestureBridge.prototype._loop = function () {
    var self = this;
    if (!this.running) return;
    this._tick();
    var interval = Math.max(60, Math.round(1000 / (this.fps || 5)));
    this._timer = setTimeout(function () {
      self._loop();
    }, interval);
  };

  GestureBridge.prototype._tick = function () {
    var v = this.video;
    if (!v || v.readyState < 2) return;
    // 上一帧还没回来就跳过这一帧：宁可掉帧，也不要让请求堆起来
    if (this._sending) return;

    var fctx = this.frameCanvas.getContext("2d");
    fctx.drawImage(v, 0, 0, this.frameCanvas.width, this.frameCanvas.height);
    this._sending = true;
    var self = this;
    this.frameCanvas.toBlob(
      function (blob) {
        if (!blob) {
          self._sending = false;
          return;
        }
        self._send(blob);
      },
      "image/jpeg",
      0.7
    );
  };

  GestureBridge.prototype._send = function (blob) {
    var self = this;
    fetch(this.endpoint, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
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
        self._sending = false;
        if (!r.ok) {
          var down = r.status === 503 || r.data.fallback === "builtin";
          self._lastError = r.data.error || "手势接口返回 " + r.status;
          self._lastReply = null;
          self._draw(null, null);
          if (down) {
            // 桥接服务没启动：停机并告诉页面切回内置算法
            self.bridgeDown = true;
            self.onStatus(
              "error",
              "本地手势服务（项目二）不可用，已停止。\n" + self._lastError
            );
            self.stop();
            if (self.onFallback) self.onFallback(self._lastError);
          } else {
            self.onStatus("error", "手势识别出错：" + self._lastError);
          }
          return;
        }
        self._lastError = "";
        self._lastReply = r.data;
        var label = r.data.gesture === "thumbs_up"
          ? "like"
          : r.data.gesture === "thumbs_down"
          ? "dislike"
          : null;
        self._draw(label, r.data);
        self._decideBridge(label);
      })
      .catch(function (err) {
        self._sending = false;
        self._lastError = err.message;
        self.onStatus("error", "无法连接本地手势服务：" + err.message);
        self.stop();
        if (self.onFallback) self.onFallback(err.message);
      });
  };

  /* 稳定保持 + 冷却：两手都要满足才触发一次反馈 */
  GestureBridge.prototype._decideBridge = function (label) {
    var now = Date.now();
    if (!label) {
      this._winLabel = null;
      this._winSince = 0;
      return;
    }
    if (label !== this._winLabel) {
      this._winLabel = label;
      this._winSince = now;
      return;
    }
    if (now - this._winSince < this.holdMs) return;
    if (now - this.lastFiredAt < this.cooldownMs) return;
    this.lastFiredAt = now;
    this._winSince = now; // 触发后要重新累计，避免一直连发
    this.lastResult = { label: label, confidence: 1 };
    this.onGesture(label === "like" ? "like" : "dislike");
  };

  /* 画布上画：当前画面 + 判定结果 + 提示（关键点由服务端算，这里不画骨骼） */
  GestureBridge.prototype._draw = function (label, reply) {
    var overlay = this.overlay;
    if (!overlay) return;
    var octx = overlay.getContext("2d");
    var w = overlay.width;
    var h = overlay.height;
    octx.clearRect(0, 0, w, h);
    if (this.video && this.video.readyState >= 2) {
      octx.drawImage(this.video, 0, 0, w, h);
    }
    var color = label === "like" ? "#22c55e" : label === "dislike" ? "#ef4444" : "#f59e0b";
    if (label) {
      octx.strokeStyle = color;
      octx.lineWidth = 4;
      octx.strokeRect(2, 2, w - 4, h - 4);
    }
    octx.font = "bold 15px sans-serif";
    octx.fillStyle = color;
    var text = label === "like" ? "👍 喜欢" : label === "dislike" ? "👎 不喜欢" : "识别中";
    octx.fillText(text, 8, 20);
    octx.font = "12px sans-serif";
    octx.fillStyle = "#e5e7eb";
    var hands = reply && typeof reply.hands === "number" ? reply.hands : 0;
    octx.fillText("MediaPipe · 检测到手 " + hands + " 只", 8, 38);
  };

  window.GestureBridge = GestureBridge;

  window.GestureRecognizer = GestureRecognizer;
})();
