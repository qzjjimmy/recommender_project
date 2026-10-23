# 数据目录说明

| 路径 | 内容 | 是否应手工覆盖 |
|------|------|----------------|
| `raw/transactions.csv` | Online Retail 原始交易数据 | 否 |
| `raw/products.csv` | 商品主数据 | 否 |
| `generated/transactions_syn.csv` | 补充生成的交易数据 | 可重新生成 |
| `generated/products_syn.csv` | 补充生成的商品数据 | 可重新生成 |
| `processed/clean_transactions.csv` | 清洗、合并后的交易表 | 由脚本生成 |
| `processed/user_sequences.csv` | 按发票切分的购买序列 | 由脚本生成 |
| `processed/recall_set.csv` | 多路融合召回集 | 由脚本生成 |
| `processed/products_all.csv` | 合并后的商品主数据 | 由脚本生成 |
| `processed/item_profile.csv` | 在线检索使用的商品画像 | 由脚本生成 |
| `processed/customer_history.csv` | 推荐首页使用的购买历史 | 由脚本生成 |
| `processed/season_stats.csv` | 季节档期统计 | 由脚本生成 |
| `processed/demo_customers.json` | 演示用户画像 | 由脚本生成 |
| `app_users.db` | 登录用户数据库 | 本地运行时数据 |
| `app_feedback.db` | 点赞/点踩反馈数据库 | 本地运行时数据 |
| `.llm_key` | 云端大模型密钥 | 本机私密配置 |

重新构建主流程时，从项目根目录依次运行 `scripts/01_clean_data.py` 至
`scripts/08_evaluate.py`。首页相关中间表由 `scripts/13_build_home_data.py` 生成。
