# 多模态个性化推荐助手

这是课程项目三的整合代码：复用项目一的推荐系统与项目二的多模态助手。项目二保持独立目录和运行环境，通过本地 HTTP 桥接服务与推荐系统通信。`integrated_ai_project` 脚手架不是运行依赖。

## 一键启动（推荐）

在 Finder 中双击根目录的 `启动项目.command`。脚本会自动完成：

1. 启动或复用本地 Ollama；
2. 用项目二环境启动或复用多模态桥接服务；
3. 启动推荐系统并打开 `/home`。

终端保持打开时项目可用，按 `Ctrl+C` 会停止本次启动的后台服务。若 macOS 首次阻止运行，请右键文件并选择“打开”。

当前 Mac 的 AirPlay 接收器默认占用 `5000`，所以一键脚本会自动改用 `5050`。启动完成后访问：

- 推荐首页：<http://127.0.0.1:5050/home>
- 智能推荐：<http://127.0.0.1:5050/search>
- 健康检查：<http://127.0.0.1:5050/health>

如需指定端口，可在终端运行：

```bash
RECOMMENDER_PORT=5060 ./启动项目.command
```

服务日志写入 `outputs/logs/`。

## 手动启动

推荐系统使用 Python 3.11。在项目根目录分别启动两个服务：

```bash
conda activate hygc_qzj
pip install -r requirements.txt
python scripts/00_check_env.py
python web/app.py
```

默认访问：

- 推荐首页：<http://127.0.0.1:5000/home>
- 智能推荐：<http://127.0.0.1:5000/search>
- 健康检查：<http://127.0.0.1:5000/health>

如果 `5000` 被 AirPlay 占用，可以直接指定其他端口：

```bash
RECOMMENDER_PORT=5050 python web/app.py
```

如需手动启用语音、手势、视觉和本机大模型能力，在另一个终端启动桥接：

```bash
cd multimodal_ai_project1
conda activate multimodal_ai
python bridge_server.py
```

桥接服务未启动时，推荐系统仍可正常使用文字搜索和基础推荐功能。

## 目录说明

| 路径 | 用途 |
|------|------|
| `config/` | 全局路径、字段和超参数 |
| `src/` | 可复用的推荐算法与业务逻辑 |
| `scripts/` | 数据清洗、训练、评测和自检入口 |
| `web/` | Flask 服务、页面模板和静态资源 |
| `data/` | 原始数据、处理中间表与本地应用数据库 |
| `models/` | Item2Vec、向量索引与精排模型 |
| `outputs/` | 按阶段编号保存的日志、表格和截图 |
| `tests/` | 自动化测试 |
| `docs/` | 推荐系统完整实践文档 |
| `multimodal_ai_project1/` | 独立的多模态 AI 助手项目 |

主流程脚本从 `scripts/00_check_env.py` 开始，依次使用 `01` 至 `08` 完成清洗、训练、召回、精排与评估。新增功能对应的自检脚本继续使用后续编号。

## 文档

- [文档索引](docs/README.md)
- [系统架构与协议](docs/architecture.md)
- [部署与风险说明](docs/deployment_and_risks.md)
- [演示与验收用例](docs/demo_cases.md)
- [完整实践文档](docs/guides/推荐系统实践_完整文档.md)
- [目录规范](目录规范.md)

## 项目三验收

六个阶段都在现有代码上提供可重跑入口，并在 `outputs/` 落盘结构化证据：

```bash
python scripts/01_contract_test.py
python scripts/02_context_test.py
python scripts/03_orchestrator_demo.py
python scripts/04_explanation_test.py
python scripts/05_feedback_ab.py
python scripts/06_e2e_acceptance.py
python scripts/19_build_submission.py
```

推荐 API 会生成或接收 `X-Trace-ID`，并让桥接服务原样返回。文字、语音、视觉三类入口
最终进入统一 `InteractionEvent` 和 Orchestrator。推荐结果包含多路召回来源和 EvidencePack；
无证据时明确降级，不生成无法核验的事实。

点赞作为长期偏好保存；点踩和继续浏览只在当前 `session_id` 生效。当前会话摘要可通过
`GET /api/session/context` 查看，撤销和清理分别使用
`POST /api/session/undo`、`POST /api/session/clear`。

根项目测试使用独立配置，避免嵌套项目的同名 `src` 包参与发现：

```bash
python -m pytest -q
```

多模态桥接测试应在 `multimodal_ai` 环境运行：

```bash
cd multimodal_ai_project1
python -m unittest discover -s tests -v
```

## 数据与密钥

`data/raw/` 中的原始数据不应被覆盖；可重新生成的数据和模型放在 `data/processed/`、`data/generated/` 与 `models/`。

账号数据库、反馈数据库和 API Key 只保存在本机，并已被 `.gitignore` 排除。云端大模型 Key 通过 `RECOMMENDER_LLM_API_KEY` 环境变量或 `data/.llm_key` 配置，不写入源码：

```bash
python scripts/11_set_llm_key.py sk-xxxx
```
