"""构建不含模型、数据库、密钥、缓存和运行日志的项目三提交包。"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS  # noqa: E402
from src.protocol import utc_now  # noqa: E402


INCLUDE_FILES = {
    "README.md",
    "requirements.txt",
    "目录规范.md",
    "启动项目.command",
    "关闭项目.command",
    "data/DATA.md",
    "data/raw/products.csv",
    "data/raw/transactions.csv",
    "docs/README.md",
    "docs/architecture.md",
    "docs/deployment_and_risks.md",
    "docs/demo_cases.md",
}
INCLUDE_DIRS = {
    "config",
    "src",
    "scripts",
    "tests",
    "web",
    "docs/guides",
}
INCLUDE_OUTPUT_GLOBS = (
    "stage01_contract_report.json",
    "stage02_context_cases.jsonl",
    "stage02_context_metrics.json",
    "stage02_error_cases.md",
    "stage03_demo.json",
    "stage03_trace_normal.json",
    "stage03_trace_empty.json",
    "stage03_trace_fallback.json",
    "stage04_explanation_cases.jsonl",
    "stage04_evidence_table.csv",
    "stage04_error_cases.md",
    "stage05_feedback_ab.json",
    "stage05_feedback_comparison.csv",
    "stage05_performance_failures.md",
    "stage06_acceptance.json",
    "stage22/stage22_project3_architecture.png",
    "stage22/stage22_vision_recommendation.json",
)
FORBIDDEN_SUFFIXES = {
    ".task",
    ".pkl",
    ".npz",
    ".model",
    ".gguf",
    ".safetensors",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pyc",
    ".pyo",
    ".log",
}


def _excluded(path: Path) -> bool:
    name = path.name
    if name in {".DS_Store", ".secret_key", ".llm_key"}:
        return True
    if "__pycache__" in path.parts or ".git" in path.parts:
        return True
    return path.suffix.lower() in FORBIDDEN_SUFFIXES


def _files() -> list[Path]:
    selected: list[Path] = []
    for relative in INCLUDE_FILES:
        path = ROOT / relative
        if path.is_file() and not _excluded(path):
            selected.append(path)
    for relative in INCLUDE_DIRS:
        base = ROOT / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not _excluded(path):
                selected.append(path)
    outputs = Path(PATHS["outputs_dir"])
    for relative in INCLUDE_OUTPUT_GLOBS:
        path = outputs / relative
        if path.is_file() and not _excluded(path):
            selected.append(path)
    report = (
        ROOT
        / "docs/reports/《行业工程实践》专升本学生实训报告（项目一至三已完成）-项目三验收补齐版.docx"
    )
    if report.is_file():
        selected.append(report)
    return sorted(set(selected), key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> int:
    files = _files()
    dist = ROOT / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    archive = dist / "project3_submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in files:
            handle.write(path, path.relative_to(ROOT).as_posix())
    manifest = {
        "built_at": utc_now(),
        "archive": str(archive.relative_to(ROOT)),
        "bytes": archive.stat().st_size,
        "files": [path.relative_to(ROOT).as_posix() for path in files],
        "exclusions": [
            "models/**",
            "data/processed/**",
            "data/generated/**",
            "data/*.db",
            "data/.llm_key",
            "data/.secret_key",
            "outputs/logs/**",
            "**/__pycache__/**",
        ],
    }
    manifest_path = Path(PATHS["outputs_dir"]) / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"提交包：{archive.relative_to(ROOT)}（{len(files)} 个文件，{archive.stat().st_size / 1024 / 1024:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
