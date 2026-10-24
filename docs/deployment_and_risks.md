# 部署与风险说明

## 环境

推荐服务使用 Python 3.10+ 和根目录 `requirements.txt`。多模态桥接使用独立环境
`multimodal_ai` 与 `multimodal_ai_project1/requirements.txt`，避免两套 `src` 和
`config` 包互相覆盖。

```bash
python -m pip install -r requirements.txt
python scripts/00_check_env.py
python web/app.py
```

多模态能力需要另开终端：

```bash
cd multimodal_ai_project1
python bridge_server.py
```

也可使用根目录 `启动项目.command`。桥接服务未启动时，页面自动保留文字和规则推荐，
视觉入口返回 `MULTIMODAL_UNAVAILABLE` 与 `fallback=text-only`。

## 数据和模型

原始交易和商品数据保留在 `data/raw/`。可重建数据、向量索引、模型和运行日志不得进入
提交包。MediaPipe 的三个 `.task` 文件只保留在本机 `models/mediapipe/`，首次使用可按
`multimodal_ai_project1/config/settings.py` 中的官方地址重新下载。

账号库、反馈库、`.secret_key` 和 `.llm_key` 只存在本机，已被 `.gitignore` 排除。

## 重跑验收

从项目根目录依次运行：

```bash
python scripts/01_contract_test.py
python scripts/02_context_test.py
python scripts/03_orchestrator_demo.py
python scripts/04_explanation_test.py
python scripts/05_feedback_ab.py
python scripts/06_e2e_acceptance.py
python scripts/19_build_submission.py
python -m pytest -q
```

阶段 6 使用确定性桥接替身覆盖语音、手势和视觉响应，不下载模型也能验证接口和故障
降级；真实模型链路由 `scripts/18_check_vision_recommend.py` 和现场演示补充。

`dist/project3_submission.zip` 是最终提交包。构建脚本使用显式清单，排除模型、数据库、
密钥、缓存、生成数据和运行日志；清单写在 `outputs/submission_manifest.json`。

## 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 多模态桥接未启动 | 语音、视觉、TTS 不可用 | `/health` 探测、明确 503、文字与规则降级 |
| VL 推理延迟高 | 图片分析等待时间长 | 独立超时、前端可编辑查询、不阻塞文字搜索 |
| 外部 TTS 网络失败 | 播报不可用 | 文本解释始终保留，错误不致命 |
| 会话进程重启 | 短期点踩和浏览丢失 | 长期点赞仍在 SQLite；新会话本来就应隔离负反馈 |
| 多轮条件冲突 | 结果不可解释 | 返回澄清问题，不静默覆盖条件 |
| 商品缺字段 | 解释可能证据不足 | EvidencePack 明确降级，不编造属性 |
| 模型文件体积大 | 提交包膨胀或泄密 | `.gitignore` 排除，部署时单独下载 |

## 隐私

系统不识别用户身份、年龄、性别、民族、健康和情绪。视觉请求只提取商品线索；上传
图片只用于当次推理，桥接临时文件在请求结束后删除。演示数据不得包含个人音视频。
