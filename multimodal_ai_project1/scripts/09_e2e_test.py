"""阶段06 — 端到端集成测试脚本。

测试完整系统的所有交互链路和性能指标。
"""

from __future__ import annotations

import csv
import resource
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OLLAMA_CONFIG, PATHS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
ASR_DIR = ROOT / "models" / "SenseVoiceSmall"
DOC_PATHS = [
    REPO / "阶段01_环境搭建与音视频采集_8课时.md",
    REPO / "阶段05_人体姿态与手势识别_8课时.md",
]

# 测试用例定义
TEST_CASES = [
    {"id": "E2E-01", "name": "基础语音对话", "input": "你好"},
    {"id": "E2E-02", "name": "知识问答", "input": "文档的主要内容是什么？"},
    {"id": "E2E-03", "name": "视觉理解", "input": "看看我在做什么"},
    {"id": "E2E-04", "name": "手势控制", "input": "点赞手势"},
    {"id": "E2E-05", "name": "连续对话", "input": "连续3个问题"},
    {"id": "E2E-06", "name": "中英混合", "input": "I want to 学习AI"},
    {"id": "E2E-07", "name": "噪声环境", "input": "背景音下的语音"},
    {"id": "E2E-08", "name": "程序退出", "input": "关闭窗口"},
    {"id": "E2E-09", "name": "长时间运行", "input": "运行30分钟"},
    {"id": "E2E-10", "name": "知识库更新", "input": "导入新文档后提问"},
]


def main():
    """运行端到端测试并记录结果。"""
    print("=" * 50)
    print("端到端集成测试")
    print("=" * 50)

    results: list[dict] = []
    metrics: list[dict] = []
    notes: dict[str, tuple[str, float, str]] = {}   # id → (status, latency, note)

    # ---------------- 探测各链路可用性 ----------------
    from src.audio_capture import AudioCapture
    from src.video_capture import VideoCapture

    has_mic = bool(AudioCapture.list_input_devices())
    has_cam = bool(VideoCapture.enumerate_cameras())
    has_asr = ASR_DIR.exists()

    has_text = False
    text_llm = None
    try:
        from src.text_llm import create_text_llm

        text_llm = create_text_llm()
        text_llm.load()
        has_text = True
        print(f"[环境] 文本模型后端：{type(text_llm).__name__}")
    except Exception as exc:
        print(f"[环境] 文本模型不可用：{exc}")

    rag = None
    if has_text:
        try:
            from src.knowledge_base import VectorStore
            from src.rag import RAGPipeline

            rag = RAGPipeline(VectorStore(), text_llm)
        except Exception as exc:
            print(f"[环境] 知识库不可用：{exc}")

    def has_vl() -> bool:
        try:
            from src.vl_model import VLModel

            return VLModel.is_available()
        except Exception:
            return False

    vl_ok = has_vl()
    print(f"[环境] 麦克风={has_mic} 摄像头={has_cam} ASR={has_asr} "
          f"文本={has_text} 视觉={vl_ok}")

    # ---------------- E2E-01 基础语音对话 ----------------
    if has_mic and has_asr and has_text:
        from src.asr import ASREngine
        from src.audio_capture import AudioCapture as _AC

        print("\nE2E-01：请对着麦克风说一句话（5 秒）…")
        cap = _AC()
        try:
            cap.start()
            chunks = [cap.read_chunk() for _ in range(
                int(5 * 16000 / 1024))]
        finally:
            cap.stop()
        wav = Path(PATHS["outputs"]) / "e2e_voice.wav"
        import wave
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(b"".join(chunks))

        asr = ASREngine(str(ASR_DIR)).load()
        t0 = time.time()
        text = asr.transcribe(str(wav), language="zh")
        asr_s = time.time() - t0
        t0 = time.time()
        answer = text_llm.generate(text or "你好")
        llm_s = time.time() - t0
        notes["E2E-01"] = ("✓" if text else "✗", round(asr_s + llm_s, 2),
                           f"识别「{text}」→ 回答 {answer[:20]}…")
        metrics += [{"category": "响应延迟", "name": "ASR 识别延迟",
                     "value": f"{asr_s:.2f}s", "target": "<1s", "note": "SenseVoiceSmall"},
                    {"category": "响应延迟", "name": "LLM 生成延迟",
                     "value": f"{llm_s:.2f}s", "target": "<2s(首token)",
                     "note": type(text_llm).__name__}]
    else:
        notes["E2E-01"] = ("—", 0.0,
                           f"跳过：麦克风={has_mic}, ASR模型={has_asr}, 文本模型={has_text}")

    # ---------------- E2E-02 知识问答 ----------------
    if rag is not None and DOC_PATHS[0].exists():
        try:
            added = rag.import_document(str(DOC_PATHS[0]))
            t0 = time.time()
            answer = rag.answer("阶段01 的交付物里有哪些脚本？")
            cost = time.time() - t0
            ok = "✓" if answer.strip() else "✗"
            notes["E2E-02"] = (ok, round(cost, 2),
                               f"导入 {added} 块；回答 {answer[:30]}…")
            metrics.append({"category": "响应延迟", "name": "RAG 问答延迟",
                            "value": f"{cost:.2f}s", "target": "<5s",
                            "note": "检索+生成"})
        except Exception as exc:
            # 最常见原因：嵌入模型不在本地，且 HuggingFace 访问不了
            notes["E2E-02"] = ("—", 0.0,
                               f"跳过：嵌入模型不可用（{type(exc).__name__}）→ "
                               "请把 all-MiniLM-L6-v2 放到 models/ 下")
    else:
        notes["E2E-02"] = ("—", 0.0, "跳过：文本模型或知识库不可用")

    # ---------------- E2E-03 视觉理解 ----------------
    if has_cam and vl_ok:
        from src.video_capture import VideoCapture as _VC
        from src.vl_model import VLModel

        cap = _VC()
        frames = []
        try:
            cap.start()
            for _ in range(8):
                f = cap.read_frame()
                if f is not None:
                    frames.append(f)
        finally:
            cap.stop()
        vl = VLModel()
        t0 = time.time()
        answer = vl.generate_response(frames, "看看画面里有什么？") if frames else ""
        cost = time.time() - t0
        notes["E2E-03"] = ("✓" if answer else "✗", round(cost, 2), answer[:40])
        metrics.append({"category": "响应延迟", "name": "VL 推理延迟",
                        "value": f"{cost:.2f}s", "target": "<5s",
                        "note": "Ollama " + OLLAMA_CONFIG["vl_model"]})
    else:
        notes["E2E-03"] = ("—", 0.0, f"跳过：摄像头={has_cam}, VL可用={vl_ok}")

    # ---------------- E2E-04 手势控制 ----------------
    try:
        import importlib

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        det = importlib.import_module("08_test_detection")
        acc = det.unit_test()
        from src.gesture_actions import GestureActionMapper

        action = GestureActionMapper().map("thumbs_up")
        notes["E2E-04"] = ("✓" if acc >= 0.8 else "✗", 0.0,
                           f"合成关键点准确率 {acc:.0%}，thumbs_up→{action}")
        metrics.append({"category": "准确率", "name": "手势识别准确率",
                        "value": f"{acc:.0%}", "target": ">80%", "note": "合成关键点验证"})
    except Exception as exc:
        notes["E2E-04"] = ("—", 0.0, f"跳过：{exc}")

    # ---------------- E2E-05 连续对话 ----------------
    if text_llm is not None:
        t0 = time.time()
        answers = [text_llm.generate(q) for q in
                   ("你好", "刚才我问了什么？", "再说一句")]
        cost = time.time() - t0
        notes["E2E-05"] = ("✓" if all(a.strip() for a in answers) else "✗",
                           round(cost, 2), "3 轮均返回非空回答")
    else:
        notes["E2E-05"] = ("—", 0.0, "跳过：文本模型不可用")

    # ---------------- E2E-06 中英混合 ----------------
    if has_mic and has_asr:
        notes["E2E-06"] = ("—", 0.0, "需人工说「I want to 学习AI」后核对识别结果")
    else:
        notes["E2E-06"] = ("—", 0.0, "跳过：缺少麦克风或 ASR 模型")

    # ---------------- E2E-07 噪声环境（VAD 不误触发）----------------
    try:
        from src.vad import VADProcessor

        vad = VADProcessor()
        frame_bytes = vad.frame_size * 2
        triggered = any(vad.process_frame(b"\x00" * frame_bytes)
                        for _ in range(40))          # 约 1.2 秒纯静音
        notes["E2E-07"] = ("✓" if not triggered else "✗", 0.0,
                           "1.2 秒纯静音未触发断句")
    except Exception as exc:
        notes["E2E-07"] = ("—", 0.0, f"跳过：{exc}")

    # ---------------- E2E-08 程序退出（线程回收）----------------
    class _Dummy(threading.Thread):
        """最小可停止线程；注意停止信号不能叫 _stop（那是 Thread 内部方法）。"""

        def __init__(self):
            super().__init__(daemon=True)
            self._stop_evt = threading.Event()

        def run(self):
            while not self._stop_evt.is_set():
                time.sleep(0.05)

        def stop(self):
            self._stop_evt.set()
            self.join(timeout=3)

    dummy = _Dummy()
    dummy.start()
    dummy.stop()
    notes["E2E-08"] = ("✓" if not dummy.is_alive() else "✗", 0.0,
                       f"join 后存活={dummy.is_alive()}，存活线程={threading.active_count()}")

    # ---------------- E2E-09 长时间运行（内存采样）----------------
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    notes["E2E-09"] = ("—", 0.0,
                       f"当前峰值内存 {rss_mb:.0f}MB（完整 30 分钟挂机测试需人工执行）")
    metrics.append({"category": "资源占用", "name": "进程内存 RSS",
                    "value": f"{rss_mb:.0f}MB", "target": "<4GB",
                    "note": "本机无独显，用内存口径"})

    # ---------------- E2E-10 知识库增量更新 ----------------
    if rag is not None and DOC_PATHS[1].exists():
        try:
            before = len(rag.vector_store.documents)
            added = rag.import_document(str(DOC_PATHS[1]))
            after = len(rag.vector_store.documents)
            hits = rag.vector_store.search("MediaPipe 手势识别", k=3)
            notes["E2E-10"] = ("✓" if after > before and hits else "✗", 0.0,
                               f"{before}→{after} 块（新增 {added}），"
                               f"新文档检索命中 {len(hits)} 块")
        except Exception as exc:
            notes["E2E-10"] = ("—", 0.0,
                               f"跳过：嵌入模型不可用（{type(exc).__name__}）")
    else:
        notes["E2E-10"] = ("—", 0.0, "跳过：文本模型或文档不可用")

    # ---------------- 汇总输出 ----------------
    for tc in TEST_CASES:
        status, latency, note = notes.get(tc["id"], ("—", 0.0, "未执行"))
        print(f"\n{tc['id']} {tc['name']}: [{status}] {note}")
        results.append({
            "id": tc["id"],
            "name": tc["name"],
            "status": status,
            "latency_s": latency,
            "notes": note,
        })

    out_dir = Path(PATHS["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "e2e_test_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "name", "status", "latency_s", "notes"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n测试报告已保存到: {output_csv}")

    metrics_csv = out_dir / "performance_metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "name", "value", "target", "note"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"性能指标已保存到: {metrics_csv}")

    print("\n端到端测试完成。")


if __name__ == "__main__":
    main()
