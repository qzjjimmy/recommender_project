/*
 * 视觉选品：网页侧只负责抓取一张图片、缩放并调用项目一视觉分析接口。
 *
 * 分工：
 *   - 浏览器：拍照/选图 → 最长边缩到 1024 → multipart 上传；
 *   - 项目一：校验登录、大小和图片格式，转发到本地桥接；
 *   - 项目二：qwen2.5vl 生成结构化场景线索；
 *   - 用户：检查并编辑推荐条件，确认后才进入现有搜索。
 */
(function () {
  "use strict";

  var MAX_SIDE = 1024;
  var MAX_UPLOAD_BYTES = 8 * 1024 * 1024;
  var STATUS_TIMEOUT_MS = 180000;

  var card = document.getElementById("vision-card");
  if (!card) return;

  var cameraBtn = document.getElementById("vision-camera-btn");
  var cameraLabel = document.getElementById("vision-camera-label");
  var fileBtn = document.getElementById("vision-file-btn");
  var fileInput = document.getElementById("vision-file");
  var cancelBtn = document.getElementById("vision-cancel");
  var preview = document.getElementById("vision-image");
  var placeholder = document.getElementById("vision-preview-placeholder");
  var statusEl = document.getElementById("vision-status");
  var resultEl = document.getElementById("vision-result");
  var sceneEl = document.getElementById("vision-scene");
  var tagsEl = document.getElementById("vision-tags");
  var queryEl = document.getElementById("vision-query");
  var useBtn = document.getElementById("vision-use");
  var engineEl = document.getElementById("vision-engine");
  var mmStatus = window.MM_STATUS || {};

  var previewUrl = "";
  var activeController = null;
  var busy = false;

  function setStatus(text, isError) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.classList.toggle("is-error", !!isError);
  }

  function setBusy(on) {
    busy = !!on;
    [cameraBtn, fileBtn, useBtn].forEach(function (button) {
      if (button) button.disabled = !!on;
    });
    if (cancelBtn) cancelBtn.hidden = !on;
    if (cameraLabel) cameraLabel.textContent = on ? "识别中…" : "拍照识别";
    if (useBtn) useBtn.classList.toggle("busy", !!on);
  }

  function renderEngine() {
    if (!engineEl) return;
    var available = !!(mmStatus.available && mmStatus.vision);
    engineEl.textContent = available
      ? "本地视觉 · " + (mmStatus.vision_model || "qwen2.5vl")
      : "视觉服务未就绪";
    engineEl.classList.toggle("is-on", available);
    engineEl.classList.toggle("is-off", !available);
  }

  function showPreview(blob) {
    if (!preview || !placeholder) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(blob);
    preview.src = previewUrl;
    preview.hidden = false;
    placeholder.hidden = true;
  }

  function imageSizeError(message) {
    setStatus(message || "无法读取这张图片，请换一张重试。", true);
  }

  function downscaleBlob(blob) {
    return new Promise(function (resolve, reject) {
      var objectUrl = URL.createObjectURL(blob);
      var img = new Image();
      img.onload = function () {
        try {
          var width = img.naturalWidth || img.width;
          var height = img.naturalHeight || img.height;
          if (!width || !height) throw new Error("图片尺寸为空");
          var scale = Math.min(1, MAX_SIDE / Math.max(width, height));
          var canvas = document.createElement("canvas");
          canvas.width = Math.max(1, Math.round(width * scale));
          canvas.height = Math.max(1, Math.round(height * scale));
          var ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(
            function (out) {
              URL.revokeObjectURL(objectUrl);
              if (out) resolve(out);
              else reject(new Error("图片压缩失败"));
            },
            "image/jpeg",
            0.85
          );
        } catch (err) {
          URL.revokeObjectURL(objectUrl);
          reject(err);
        }
      };
      img.onerror = function () {
        URL.revokeObjectURL(objectUrl);
        reject(new Error("无法解码图片"));
      };
      img.src = objectUrl;
    });
  }

  function captureCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("当前浏览器不支持摄像头拍摄，请改用「选择图片」。", true);
      return Promise.reject(new Error("camera unsupported"));
    }
    var attempts = [
      {
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      },
      {
        video: {
          facingMode: { ideal: "user" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      },
      { video: true, audio: false },
    ];

    function tryAt(index, lastError) {
      if (index >= attempts.length) {
        return Promise.reject(lastError || new Error("无法打开摄像头"));
      }
      return navigator.mediaDevices.getUserMedia(attempts[index])
        .catch(function (err) { return tryAt(index + 1, err); });
    }

    setStatus("正在请求摄像头权限并抓取画面…");
    return tryAt(0).then(function (stream) {
      var video = document.createElement("video");
      video.playsInline = true;
      video.muted = true;
      video.srcObject = stream;
      return video.play().then(function () {
        return new Promise(function (resolve, reject) {
          window.setTimeout(function () {
            try {
              var width = video.videoWidth;
              var height = video.videoHeight;
              if (!width || !height) {
                throw new Error("摄像头没有返回有效画面");
              }
              var scale = Math.min(1, MAX_SIDE / Math.max(width, height));
              var canvas = document.createElement("canvas");
              canvas.width = Math.max(1, Math.round(width * scale));
              canvas.height = Math.max(1, Math.round(height * scale));
              canvas.getContext("2d").drawImage(
                video, 0, 0, canvas.width, canvas.height);
              canvas.toBlob(function (blob) {
                if (blob) resolve(blob);
                else reject(new Error("摄像头画面转换失败"));
              }, "image/jpeg", 0.85);
            } catch (err) {
              reject(err);
            }
          }, 180);
        });
      }).finally(function () {
        stream.getTracks().forEach(function (track) { track.stop(); });
        video.srcObject = null;
      });
    });
  }

  function renderResult(data) {
    var vision = data.vision || {};
    if (sceneEl) {
      sceneEl.textContent = vision.scene || "画面中没有提取到稳定描述。";
    }
    if (tagsEl) {
      tagsEl.innerHTML = "";
      var groups = [
        { label: "类目", values: vision.category_hints || [], category: true },
        { label: "物体", values: vision.objects || [], category: false },
        { label: "颜色", values: vision.colors || [], category: false },
      ];
      groups.forEach(function (group) {
        group.values.forEach(function (value) {
          var tag = document.createElement("span");
          tag.className = "vision-tag" + (group.category ? " is-category" : "");
          tag.textContent = group.label + " · " + value;
          tagsEl.appendChild(tag);
        });
      });
    }
    if (queryEl) queryEl.value = data.suggested_query || "";
    if (resultEl) resultEl.hidden = false;
    var seconds = data.elapsed_ms ? (data.elapsed_ms / 1000).toFixed(1) : "";
    setStatus(
      "识别完成" + (seconds ? "，耗时 " + seconds + " 秒" : "") +
      "。请核对描述并确认搜索条件。",
      false
    );
    if (useBtn) useBtn.disabled = !(queryEl && queryEl.value.trim());
  }

  function analyze(blob) {
    if (busy) return;
    if (!blob) {
      imageSizeError("没有取得有效图片。");
      return;
    }
    if (blob.size > MAX_UPLOAD_BYTES) {
      imageSizeError("压缩后的图片仍超过 8MB，请换一张重试。");
      return;
    }

    showPreview(blob);
    if (resultEl) resultEl.hidden = true;
    setBusy(true);
    setStatus("正在调用本地视觉模型；首次请求可能需要加载模型，请稍候…");

    var controller = new AbortController();
    activeController = controller;
    var timer = window.setTimeout(function () { controller.abort(); }, STATUS_TIMEOUT_MS);
    var form = new FormData();
    form.append("image", blob, "vision.jpg");

    fetch("/api/vision/analyze", {
      method: "POST",
      body: form,
      signal: controller.signal,
    })
      .then(function (resp) {
        return resp.json()
          .catch(function () { return {}; })
          .then(function (data) { return { ok: resp.ok, status: resp.status, data: data }; });
      })
      .then(function (reply) {
        if (!reply.ok) {
          var hint = reply.data.hint || reply.data.error || ("视觉分析失败：" + reply.status);
          throw new Error(hint);
        }
        mmStatus.available = true;
        mmStatus.vision = true;
        renderEngine();
        renderResult(reply.data);
      })
      .catch(function (err) {
        var message = err && err.name === "AbortError"
          ? "识别已取消或超时，文字与语音搜索仍可继续使用。"
          : ("视觉分析失败：" + (err.message || "未知错误"));
        setStatus(message, true);
      })
      .finally(function () {
        window.clearTimeout(timer);
        activeController = null;
        setBusy(false);
        if (queryEl) useBtn.disabled = !queryEl.value.trim();
      });
  }

  if (cameraBtn) {
    cameraBtn.addEventListener("click", function () {
      captureCamera()
        .then(function (blob) {
          if (!blob) throw new Error("摄像头画面转换失败");
          analyze(blob);
        })
        .catch(function (err) {
          var name = (err && err.name) || "";
          var text = name === "NotAllowedError"
            ? "摄像头权限被拒绝，请在浏览器地址栏允许权限，或改用「选择图片」。"
            : name === "NotFoundError"
              ? "没有检测到摄像头，请改用「选择图片」。"
              : (err.message || "摄像头不可用");
          imageSizeError(text);
        });
    });
  }

  if (fileBtn && fileInput) {
    fileBtn.addEventListener("click", function () { fileInput.click(); });
    fileInput.addEventListener("change", function () {
      var file = fileInput.files && fileInput.files[0];
      fileInput.value = "";
      if (!file) return;
      setStatus("正在压缩图片…");
      downscaleBlob(file)
        .then(analyze)
        .catch(function (err) { imageSizeError(err.message); });
    });
  }

  if (cancelBtn) {
    cancelBtn.addEventListener("click", function () {
      if (activeController) activeController.abort();
    });
  }

  if (queryEl && useBtn) {
    queryEl.addEventListener("input", function () {
      useBtn.disabled = busy || !queryEl.value.trim();
    });
  }

  if (useBtn && queryEl) {
    useBtn.addEventListener("click", function () {
      var query = queryEl.value.trim();
      if (!query) {
        setStatus("请先填写至少一个推荐条件。", true);
        queryEl.focus();
        return;
      }
      window.location.href = "/search?from=vision&q=" + encodeURIComponent(query);
    });
  }

  window.addEventListener("beforeunload", function () {
    if (activeController) activeController.abort();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });

  renderEngine();
})();
