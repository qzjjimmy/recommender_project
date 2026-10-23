/* 设备检测工具：在真正调用摄像头/麦克风之前，先判断设备存不存在。
 *
 * 为什么要先检测：
 *   如果没有设备就直接调 getUserMedia，浏览器会抛 NotFoundError，
 *   而这个错误很容易被误读成"权限被拒绝"（Chrome 的语音识别在没有麦克风时
 *   就报 not-allowed），导致用户去反复检查权限设置，白折腾。
 *   先 enumerateDevices() 判断设备种类是否存在，就能给出准确的提示。
 *
 * 注意：未授权时 enumerateDevices() 返回的 label 是空字符串，但**设备数量与种类是真实的**，
 * 这正是我们需要的。
 */
(function () {
  "use strict";

  var KIND_LABEL = {
    videoinput: "摄像头",
    audioinput: "麦克风",
    audiooutput: "扬声器/耳机",
  };

  function supported() {
    return !!(window.navigator.mediaDevices && navigator.mediaDevices.enumerateDevices);
  }

  function list() {
    if (!supported()) return Promise.resolve([]);
    return navigator.mediaDevices.enumerateDevices().catch(function () {
      return [];
    });
  }

  function has(kind) {
    return list().then(function (devices) {
      return devices.some(function (d) {
        return d.kind === kind;
      });
    });
  }

  function summary() {
    return list().then(function (devices) {
      var counts = {};
      devices.forEach(function (d) {
        counts[d.kind] = (counts[d.kind] || 0) + 1;
      });
      return {
        total: devices.length,
        counts: counts,
        items: devices.map(function (d) {
          return {
            kind: d.kind,
            kindLabel: KIND_LABEL[d.kind] || d.kind,
            label: d.label || "",
          };
        }),
        hasCamera: !!counts.videoinput,
        hasMicrophone: !!counts.audioinput,
      };
    });
  }

  /* 把 getUserMedia 的异常翻译成能直接指导操作的中文，
     并区分"没有设备"与"权限被拒"这两件完全不同的事。 */
  function explainError(err, kindLabel) {
    var name = (err && err.name) || "UnknownError";
    var map = {
      NotFoundError:
        "没有检测到" + kindLabel + "。请确认设备已连接并在系统里被识别（如果是笔记本，检查设备是否被禁用）。",
      NotAllowedError:
        "浏览器拒绝了" + kindLabel + "访问。请点击地址栏左侧的图标允许访问，然后刷新页面重试。",
      PermissionDeniedError:
        "浏览器拒绝了" + kindLabel + "访问。请点击地址栏左侧的图标允许访问，然后刷新页面重试。",
      NotReadableError:
        kindLabel + "被其它程序占用（例如另一个浏览器标签页或会议软件）。请关闭它们后重试。",
      TrackStartError: kindLabel + "启动失败，可能被其它程序占用。",
      OverconstrainedError:
        kindLabel + "不支持所要求的参数。已尝试降级，若仍失败请更换设备。",
      SecurityError:
        "当前页面不是安全上下文，浏览器禁止访问" + kindLabel + "。请用 http://127.0.0.1 或 https 打开。",
      AbortError: kindLabel + "启动被中断，请重试。",
    };
    return map[name] || "无法访问" + kindLabel + "：" + (err && err.message ? err.message : name);
  }

  window.DeviceUtil = {
    KIND_LABEL: KIND_LABEL,
    supported: supported,
    list: list,
    has: has,
    summary: summary,
    explainError: explainError,
  };
})();
