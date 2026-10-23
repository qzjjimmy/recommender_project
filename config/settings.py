"""全局配置：路径、随机种子、常用超参。

学生可改文件名/超参，但请保持 SEED=42，便于组间对照。
"""
from __future__ import annotations

from pathlib import Path

# 项目根目录（无论从哪启动脚本，都能定位 data/ models/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42

PATHS = {
    "raw_transactions": PROJECT_ROOT / "data" / "raw" / "transactions.csv",
    "raw_products": PROJECT_ROOT / "data" / "raw" / "products.csv",
    "clean_transactions": PROJECT_ROOT / "data" / "processed" / "clean_transactions.csv",
    "user_sequences": PROJECT_ROOT / "data" / "processed" / "user_sequences.csv",
    "recall_set": PROJECT_ROOT / "data" / "processed" / "recall_set.csv",
    # 合并后的商品主数据（原始 + 补充），由 scripts/01_clean_data.py 生成；
    # 下游统一读它，读不到时自动回退到 data/raw/products.csv
    "products_all": PROJECT_ROOT / "data" / "processed" / "products_all.csv",
    "models_dir": PROJECT_ROOT / "models",
    "vector_index_dir": PROJECT_ROOT / "models" / "vector_index",
    "item2vec_model": PROJECT_ROOT / "models" / "item2vec.model",
    "outputs_dir": PROJECT_ROOT / "outputs",
    # --- 应用层（登录 / 反馈 / 智能推荐）新增 ---
    "user_db": PROJECT_ROOT / "data" / "app_users.db",
    "feedback_db": PROJECT_ROOT / "data" / "app_feedback.db",
    "item_profile": PROJECT_ROOT / "data" / "processed" / "item_profile.csv",
    "demo_customers": PROJECT_ROOT / "data" / "processed" / "demo_customers.json",
    "llm_key_file": PROJECT_ROOT / "data" / ".llm_key",  # 大模型 API Key（已 gitignore，不入库）
    # --- 推荐首页（购买相似 + 季节档期）新增 ---
    "customer_history": PROJECT_ROOT / "data" / "processed" / "customer_history.csv",
    "season_stats": PROJECT_ROOT / "data" / "processed" / "season_stats.csv",
    # --- 补充数据（scripts/12_generate_synthetic_data.py 生成；原始数据不动）---
    "generated_dir": PROJECT_ROOT / "data" / "generated",
    # 额外的数据源：清洗时与原始数据用同一套规则分别清洗，再合并供后续阶段使用。
    # 想再加数据源往列表里追加路径即可，代码不用改。
    "extra_transactions": [PROJECT_ROOT / "data" / "generated" / "transactions_syn.csv"],
    "extra_products": [PROJECT_ROOT / "data" / "generated" / "products_syn.csv"],
}


def stage_output_dir(stage: int | str) -> Path:
    """Return (and create) the output directory for one pipeline stage.

    Accepts ``1``, ``"01"`` or ``"stage01"`` so scripts and maintenance tools
    can use the same convention without scattering path construction.
    """
    value = f"{stage:02d}" if isinstance(stage, int) else str(stage).strip()
    if value.isdigit():
        name = f"stage{int(value):02d}"
    elif value.startswith("stage") and value[5:].isdigit():
        name = f"stage{int(value[5:]):02d}"
    else:
        raise ValueError(f"invalid stage: {stage!r}")
    target = PROJECT_ROOT / "outputs" / name
    target.mkdir(parents=True, exist_ok=True)
    return target


# 字段约定（若你改了原始列名，只改这里）
COLUMNS = {
    "invoice": "InvoiceNo",
    "item": "StockCode",
    "user": "CustomerID",
    "time": "InvoiceDate",
    "qty": "Quantity",
    "price": "UnitPrice",
    "country": "Country",
    "desc": "Description",
}

HYPERPARAMS = {
    "item2vec": {
        "vector_size": 64,
        "window": 5,
        "min_count": 5,
        "sg": 1,  # Skip-gram
        "epochs": 10,
        "negative": 5,  # 负采样个数
        "sample": 0.001,  # 高频词下采样阈值
        "workers": 1,  # 1 = 完全可复现（workers>1 时多线程累加顺序不定）
    },
    "sequence": {
        "split_by": "invoice",  # 课堂推荐：同一发票内的商品构成一条序列
        "dedupe": True,  # 同一发票内重复出现的商品只保留首次出现
        "sample_size": 3,  # 固定种子抽样条数（人工判读用）
    },
    "recall": {
        "final_k": 50,
        "quotas": {"popular": 10, "similar": 20, "user_vec": 20},
        "quotas_variant": {"popular": 5, "similar": 15, "user_vec": 30},  # 配额对比用：更偏个性化
        "recent_n": 50,  # 用户向量只取最近 N 个历史商品（None = 全部历史）
        "similar_history_n": 20,  # 相似扩展路只用最近 N 个历史商品做扩展
        "similar_per_item": 5,  # 相似扩展路每个历史商品取几个相似商品
        "hot_top_n": 100,  # "热门商品"定义：训练期交互次数 Top N
    },
    "cf": {
        "top_similar": 200,  # 每个商品保留最相似的 N 个邻居（0 = 不截断）
    },
    "eval": {
        "valid_ratio": 0.2,  # 按时间切分，时间后 20% 作为验证期
        "k": 10,  # 评测截断长度
        "sample_users": 200,  # 固定种子抽样的评测用户数
    },
    "rank": {
        "topn": 10,
        "lookback_days": 180,  # 特征的固定回看窗口；训练与评测用同样长度，保证口径可比
        "inner_label_ratio": 0.3,  # 训练期内再切分：后 30% 作标签期（特征只能用更早的数据）
        # 负样本来源：与推理同构的**召回候选池**（不用全库采样，避免训练/推理分布偏移）。
        # negative_ratio 是"每个正样本最多保留几个负样本"；0 = 保留池内全部候选，与线上一致。
        "negative_ratio": 0,
        "max_train_users": 2000,  # 参与训练样本构造的用户数上限
        "lr": {"C": 1.0, "max_iter": 1000},
        "lgbm": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
        },
    },
    "search": {
        "top_n": 24,  # 智能推荐一次返回多少条
        # 排序权重：个性化相似度 / 商品热度 / 反馈偏好
        "weights": {"similarity": 0.50, "popularity": 0.25, "feedback": 0.25},
        "feedback_weight": 0.8,  # 点赞向量在合成用户向量里的权重
        "dislike_penalty": 0.35,  # 与点踩商品相似的候选要被扣多少分
        "max_price_scan": 100000,  # 价格上限的合法性保护
        # 大模型查询解析（规则搞不定的模糊表达才调用）。
        # Key 的读取优先级：环境变量 RECOMMENDER_LLM_API_KEY
        #                  > data/.llm_key（scripts/11_set_llm_key.py 写入）
        #                  > 下面的 api_key 默认值
        "llm": {
            "enabled": True,
            # 查询解析走哪个后端：
            #   "bridge" = 项目二（multimodal_ai_project1）的本地桥接服务，
            #              用本地 Ollama 模型解析，不联网、不要 API Key（推荐）；
            #   "openai" = 下面 base_url 指向的任意 OpenAI 兼容服务（如 DeepSeek）。
            # 桥接服务不可用时两种后端都会静默退回"纯规则解析"，页面不会报错。
            "backend": "bridge",
            "base_url": "https://api.deepseek.com/v1",  # 任意 OpenAI 兼容服务
            "model": "deepseek-chat",
            # 不在源码里保存密钥。优先级：
            # RECOMMENDER_LLM_API_KEY > data/.llm_key（推荐）
            "api_key": "",
            "timeout": 8,
            "max_price": 100000,
        },
        # --- 项目二多模态桥接服务（本地 ASR / 手势 / 视觉 / 大模型 / TTS）---
        # 由 multimodal_ai_project1/bridge_server.py 提供，跑在项目二的 conda 环境里：
        #     cd multimodal_ai_project1 && python bridge_server.py
        # 服务没启动时，页面的语音/手势/播报会自动降级到项目一原有实现；
        # 视觉选品明确提示不可用，文字搜索不受影响。
        "multimodal": {
            "enabled": True,
            "base_url": "http://127.0.0.1:8100",
            # 各接口的独立超时（秒）：ASR/视觉/TTS/大模型都慢，健康探测必须快。
            "timeout": {
                "health": 1.5,
                "asr": 60,
                "gesture": 6,
                "vision": 180,
                "tts": 45,
                "llm": 180,
            },
            "health_ttl": 5.0,  # 健康探测结果缓存多少秒，避免每个请求都探一次
            "gesture_fps": 5,  # 浏览器每秒送几帧给 MediaPipe
            "asr_sample_rate": 16000,  # 语音识别输入采样率（SenseVoiceSmall 要 16k）
            "vision_max_upload_bytes": 8 * 1024 * 1024,
            "vision_browser_max_side": 1024,
        },
    },
    "home": {
        # ---- 板块一「猜你还会买」：基于购买过的商品做相似扩展 ----
        "similar_history_n": 20,  # 只用最近 N 个历史商品做扩展（与阶段四 same 口径）
        "similar_per_item": 12,  # 每个历史商品取 TopN 个相似商品
        "similar_top_n": 6,  # 首页这个板块展示几条
        "recency_decay": 0.92,  # 越早买过的商品权重越低（0.92**序号）
        "category_cap": 2,  # 同一类目在一排卡片里最多占几个位置（避免近重复刷屏）
        "weights_similar": {"similarity": 0.75, "popularity": 0.25},
        # ---- 板块二「季节档期」：按当月档期推类目 ----
        # 每个档期展示几条：按类目轮流取，10 条能覆盖 10 个品类（开学季 = 数码 + 文教 + 宿舍）
        "season_top_n": 10,
        "max_themes": 2,  # 一个月最多展示几个档期
        "weights_season": {"season": 0.45, "popularity": 0.30, "personal": 0.25},
        "season_lift_clip": 3.0,  # 季节指数截断：类目太小众时不让它一骑绝尘
        # 类目当月成交行数不足这个数时，季节指数按 1.0（无信号）处理。
        # 60 行是"至少几十笔真实成交"的量级：像"饰品"这种全库只有 41 个商品、
        # 当月 40 行的类目，指数波动纯属噪声，不该出现在首页的类目证据里。
        "season_min_events": 60,
        "season_keyword_score": 0.6,  # 关键词命中档期（但类目不在此档期池）时的基础分
        # ---- 顶部滚动推荐（轮播）：把两个板块的头部商品混成一条"最显眼"的带子 ----
        "carousel_slides": 5,  # 轮播几条（建议 4~6：再多就要等很久才转回来）
        "carousel_interval_ms": 6000,  # 自动滚动的间隔（毫秒）；前端在悬停/切到后台时暂停
        "carousel_max_per_category": 1,  # 同一个类目最多占几条（照片是按类目取的，重复就没变化）
    },
}


def set_seed(seed: int = SEED) -> None:
    """各脚本开头调用，尽量保证可复现。"""
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
