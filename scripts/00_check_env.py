#!/usr/bin/env python3
"""环境与数据自检（脚手架可直接运行，不依赖业务实现）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PATHS, SEED, set_seed  # noqa: E402


def main() -> None:
    set_seed()
    print(f"SEED={SEED}")
    print(f"PROJECT_ROOT={ROOT}")

    missing = []
    for key in ("raw_transactions", "raw_products"):
        p = PATHS[key]
        ok = p.exists()
        print(f"[{'OK' if ok else 'MISSING'}] {key}: {p}")
        if not ok:
            missing.append(str(p))

    for dkey in ("models_dir", "outputs_dir"):
        PATHS[dkey].mkdir(parents=True, exist_ok=True)
        print(f"[OK] dir ready: {PATHS[dkey]}")

    if missing:
        print("原始数据缺失，请向教师索取或检查 data/raw/。")
        sys.exit(1)

    # 轻量读几行，确认编码
    import pandas as pd

    tx = pd.read_csv(PATHS["raw_transactions"], nrows=3, dtype=str)
    pr = pd.read_csv(PATHS["raw_products"], nrows=3, dtype=str)
    print("transactions columns:", list(tx.columns))
    print("products columns:", list(pr.columns))
    print("自检通过。接下来按阶段任务书实现 src/ 与 scripts/。")


if __name__ == "__main__":
    main()
