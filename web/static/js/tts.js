/* 语音播报：把当前页的推荐结果念出来。
 *
 * 走项目二的 edge-tts（微软神经语音）：文本 → POST /api/tts → 项目一转发给
 * 桥接服务合成 MP3 → 浏览器播放。**手动触发**，不自动播放，避免打扰。
 *
 * 桥接服务不可用时不会静默失败：状态栏会说明原因（并且不伪造"已播报"）。
 */
(function () {
  "use strict";

  function noop() {}

  function TTSPlayer(options) {
    options = options || {};
    this.endpoint = options.endpoint || "/api/tts";
    this.onStatus = options.onStatus || noop;
    this.audio = null;
    this.url = null;
    this.playing = false;
  }

  TTSPlayer.prototype.supported = function () {
    return typeof window.fetch === "function" && typeof window.Audio === "function";
  };

  /* 播放一段文本；返回 Promise<boolean>（true 表示真的开始播放了）。 */
  TTSPlayer.prototype.speak = function (text) {
    var self = this;
    text = (text || "").trim();
    if (!text) {
      this.onStatus("error", "没有可播报的内容");
      return Promise.resolve(false);
    }
    if (!this.supported()) {
      this.onStatus("error", "当前浏览器不支持音频播放");
      return Promise.resolve(false);
    }
    this.stop();
    this.onStatus("synthesizing", "正在合成语音（项目二 edge-tts）…");
    return fetch(this.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    })
      .then(function (resp) {
        if (!resp.ok) {
          return resp
            .json()
            .catch(function () {
              return {};
            })
            .then(function (data) {
              throw new Error(data.error || "语音合成失败（HTTP " + resp.status + "）");
            });
        }
        return resp.blob();
      })
      .then(function (blob) {
        var url = URL.createObjectURL(blob);
        var audio = new Audio(url);
        self.url = url;
        self.audio = audio;
        audio.onended = function () {
          self.stop();
          self.onStatus("done", "播报结束");
        };
        audio.onerror = function () {
          self.stop();
          self.onStatus("error", "音频播放失败，请检查系统音量或输出设备");
        };
        return audio.play().then(
          function () {
            self.playing = true;
            self.onStatus("playing", "正在播报当前页推荐结果（点按钮可停止）");
            return true;
          },
          function (err) {
            self.stop();
            self.onStatus("error", "浏览器阻止了自动播放：" + (err && err.message ? err.message : "请再点一次"));
            return false;
          }
        );
      })
      .catch(function (err) {
        self.onStatus("error", err.message || "语音播报失败");
        return false;
      });
  };

  TTSPlayer.prototype.stop = function () {
    if (this.audio) {
      try {
        this.audio.pause();
      } catch (e) {
        /* 忽略 */
      }
      this.audio = null;
    }
    if (this.url) {
      URL.revokeObjectURL(this.url);
      this.url = null;
    }
    this.playing = false;
  };

  /* 点一下开始，再点一下停止。 */
  TTSPlayer.prototype.toggle = function (text) {
    if (this.playing) {
      this.stop();
      this.onStatus("stopped", "已停止播报");
      return Promise.resolve(false);
    }
    return this.speak(text);
  };

  /* 把推荐结果列表拼成一段自然的话（网页和语音的措辞不一样）。 */
  TTSPlayer.prototype.buildSummary = function (rows, limit) {
    limit = limit || 5;
    var parts = [];
    for (var i = 0; i < rows.length && parts.length < limit; i++) {
      var name = (rows[i].name || "").trim();
      if (!name) continue;
      var price = rows[i].price ? "，价格 " + rows[i].price + " 元" : "";
      var reason = (rows[i].reason || "").trim();
      parts.push("第 " + (parts.length + 1) + " 个，" + name + price + (reason ? "，推荐理由：" + reason : ""));
    }
    if (!parts.length) return "";
    return "为你推荐 " + parts.length + " 件商品。" + parts.join("。") + "。";
  };

  window.TTSPlayer = TTSPlayer;
})();
