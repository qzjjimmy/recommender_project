/* 语音输入：把中文语音识别成文字，填进查询框。
 *
 * 用浏览器内置的 Web Speech API（Chrome / Edge 的 webkitSpeechRecognition），
 * 不依赖任何外部服务与模型文件。识别不在本机完成，而是由浏览器厂商的
 * 在线服务处理，因此**没网时会失败** —— 失败时给出明确提示，让用户改用键盘输入。
 *
 * 注意：getUserMedia / SpeechRecognition 都要求安全上下文，
 * 本项目通过 http://127.0.0.1 访问，属于安全上下文，可以正常使用。
 */
(function () {
  "use strict";

  function VoiceInput(options) {
    options = options || {};
    this.lang = options.lang || "zh-CN";
    this.onResult = options.onResult || function () {};
    this.onStatus = options.onStatus || function () {};
    this.onError = options.onError || function () {};

    var Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognizer = Ctor ? new Ctor() : null;
    this.listening = false;
    this._gotEvent = false; // 是否收到过任何事件（用于识别"API 存在但完全不工作"的情况）

    if (this.recognizer) {
      this.recognizer.lang = this.lang;
      this.recognizer.continuous = false;
      this.recognizer.interimResults = true;
      this.recognizer.maxAlternatives = 1;
      this._bind();
    }
  }

  VoiceInput.prototype.supported = function () {
    return !!this.recognizer;
  };

  VoiceInput.prototype._bind = function () {
    var self = this;

    this.recognizer.onstart = function () {
      self._gotEvent = true;
      self.listening = true;
      self.onStatus("listening", "正在听…请说出你的需求");
    };

    this.recognizer.onresult = function (event) {
      var text = "";
      var isFinal = false;
      for (var i = event.resultIndex; i < event.results.length; i++) {
        text += event.results[i][0].transcript;
        if (event.results[i].isFinal) isFinal = true;
      }
      self.onResult(text.trim(), isFinal);
    };

    this.recognizer.onerror = function (event) {
      self._gotEvent = true;
      self.listening = false;
      var map = {
        // 注意：没有麦克风设备时 Chrome 也会报 not-allowed，
        // 所以这里不能一口咬定是权限问题。设备预检如果及时返回了结果，
        // 就借此把话说得更准；没返回就保留"两种可能都提示到"的原话。
        "not-allowed": self._noDevice
          ? "没有检测到麦克风设备。语音输入需要麦克风；" +
            "请插上麦克风后点「检测设备」，或直接改用键盘输入。"
          : "无法使用麦克风：可能是权限被拒绝，也可能是电脑没有麦克风设备。" +
            "请点「检测设备」确认，或直接改用键盘输入。",
        "service-not-allowed": "浏览器禁止了语音识别服务",
        "no-speech": "没有听到声音，请靠近麦克风再说一次",
        "audio-capture": "找不到麦克风设备，请确认已插入麦克风",
        network: "语音识别需要联网，当前网络不可用，请改用键盘输入",
        aborted: "已取消",
      };
      // 标记"这次是出错结束"，让紧随其后的 onend 不要用"已结束"把原因盖掉
      self._errored = true;
      self.onStatus("error", map[event.error] || "语音识别出错：" + event.error);
      self.onError(event.error);
    };

    this.recognizer.onend = function () {
      self._gotEvent = true;
      self.listening = false;
      // 识别失败时，浏览器同样会再抛一次 onend。如果这里照常态发"语音输入已结束"，
      // 就会把刚显示出来的失败原因覆盖掉，用户只看到一句"结束了"，
      // 完全不知道是没麦克风、没权限还是网络不通 —— 观感就是"点了没反应"。
      if (self._errored) return;
      self.onStatus("ended", "语音输入已结束");
    };
  };

  VoiceInput.prototype.start = function () {
    var self = this;
    if (!this.recognizer) {
      this.onStatus("error", "当前浏览器不支持语音识别，请改用键盘输入（推荐 Chrome / Edge）");
      return false;
    }
    if (this.listening) return true;

    this._gotEvent = false;
    this._noDevice = false;
    this._errored = false;

    // 设备预检只用来"把提示说得更准"，绝不能挡住启动：
    //   ① enumerateDevices() 在少数环境下会一直不返回；一旦等它，
    //      recognizer.start() 永远不会被调用，按钮就成了"点了没反应"；
    //   ② 麦克风权限被设成「阻止」时，设备列表里看不到音频输入设备，
    //      但识别本身还能用（浏览器会自己给出准确的错误）。
    // 所以这里异步跑预检，结果只用于解释后面的报错，不参与"要不要启动"的判断。
    if (window.DeviceUtil) {
      try {
        Promise.resolve(DeviceUtil.has("audioinput")).then(function (exists) {
          self._noDevice = !exists;
        }).catch(function () {
          /* 预检失败不影响启动 */
        });
      } catch (err) {
        /* 同上：预检自身出错不影响启动 */
      }
    }

    try {
      this.recognizer.start();
    } catch (err) {
      // 连续调用 start() 会抛 InvalidStateError，忽略即可
      this._gotEvent = true;
      this.onStatus("error", "启动语音识别失败：" + err.message);
      return false;
    }

    // 看门狗：有些浏览器 SpeechRecognition 对象存在、start() 也不报错，
    // 但一个事件都不触发（等于功能是空的）。这种情况必须主动告诉用户，
    // 否则界面会一直卡在"没反应"，让人以为是权限或设备问题。
    clearTimeout(this._watchdog);
    this._watchdog = setTimeout(function () {
      if (self._gotEvent) return;
      self.onStatus(
        "unsupported",
        "这个浏览器没有响应语音识别（对象存在但完全不工作）。\n" +
          "解决办法：① 用 Chrome 或 Edge 打开 http://127.0.0.1:5000/ 再试（推荐）；\n" +
          "② 或点一下输入框，按 Win + H 用 Windows 自带的语音输入说需求，说完点「开始推荐」。"
      );
    }, 2500);
    return true;
  };

  VoiceInput.prototype.stop = function () {
    if (this.recognizer && this.listening) {
      try {
        this.recognizer.stop();
      } catch (err) {
        /* 忽略 */
      }
    }
  };

  VoiceInput.prototype.toggle = function () {
    if (this.listening) {
      this.stop();
      return false;
    }
    return this.start();
  };

  window.VoiceInput = VoiceInput;
})();
