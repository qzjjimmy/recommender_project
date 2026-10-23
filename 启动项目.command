#!/bin/zsh

# 一键启动推荐系统、项目二多模态桥接和本地 Ollama。
# 双击运行；终端保持打开时项目可用，按 Ctrl+C 停止本次启动的服务。

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

APP_ENV="hygc_qzj"
BRIDGE_ENV="multimodal_ai"
BRIDGE_PORT=8100
OLLAMA_URL="http://127.0.0.1:11434"
BRIDGE_URL="http://127.0.0.1:${BRIDGE_PORT}"
LOG_DIR="$PROJECT_ROOT/outputs/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
APP_LOG="$LOG_DIR/recommender_${STAMP}.log"
BRIDGE_LOG="$LOG_DIR/multimodal_bridge_${STAMP}.log"
OLLAMA_LOG="$LOG_DIR/ollama_${STAMP}.log"

BRIDGE_PID=""
OLLAMA_PID=""
BROWSER_WAIT_PID=""
BRIDGE_OWNED=0
OLLAMA_OWNED=0

die() {
  echo
  echo "[启动失败] $1" >&2
  echo "日志目录：$LOG_DIR" >&2
  exit 1
}

http_ready() {
  /usr/bin/curl -fsS --max-time 2 "$1" >/dev/null 2>&1
}

port_is_busy() {
  /usr/sbin/lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

terminate_owned_process() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" >/dev/null 2>&1 || return 0
  kill -TERM "$pid" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    kill -0 "$pid" >/dev/null 2>&1 || break
    sleep 0.1
  done
  kill -0 "$pid" >/dev/null 2>&1 && kill -KILL "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
}

cleanup() {
  trap - EXIT INT TERM HUP
  echo
  echo "[清理] 正在停止本次启动的后台服务..."
  if [[ -n "$BROWSER_WAIT_PID" ]]; then
    terminate_owned_process "$BROWSER_WAIT_PID"
  fi
  if [[ "$BRIDGE_OWNED" -eq 1 ]]; then
    terminate_owned_process "$BRIDGE_PID"
  fi
  if [[ "$OLLAMA_OWNED" -eq 1 ]]; then
    terminate_owned_process "$OLLAMA_PID"
  fi
  echo "[清理] 完成。"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

echo "=============================================="
echo " 电商商品智能推荐系统 · 一键启动"
echo "=============================================="
echo "项目目录：$PROJECT_ROOT"

mkdir -p "$LOG_DIR"

CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || die "找不到 Conda 初始化脚本：$CONDA_SH"
source "$CONDA_SH"
command -v conda >/dev/null 2>&1 || die "Conda 未安装或未正确初始化"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
[[ -n "$CONDA_BASE" ]] || die "无法读取 Conda 安装目录"

APP_PYTHON="$CONDA_BASE/envs/$APP_ENV/bin/python"
BRIDGE_PYTHON="$CONDA_BASE/envs/$BRIDGE_ENV/bin/python"

[[ -x "$APP_PYTHON" ]] || die "找不到 Conda 环境 $APP_ENV"
[[ -x "$BRIDGE_PYTHON" ]] || die "找不到 Conda 环境 $BRIDGE_ENV"
[[ -f "$PROJECT_ROOT/web/app.py" ]] || die "缺少推荐系统入口 web/app.py"
[[ -f "$PROJECT_ROOT/multimodal_ai_project1/bridge_server.py" ]] \
  || die "缺少多模态桥接入口 bridge_server.py"

APP_HOST="${RECOMMENDER_HOST:-127.0.0.1}"
if [[ -n "${RECOMMENDER_PORT:-}" ]]; then
  APP_PORT="$RECOMMENDER_PORT"
  [[ "$APP_PORT" =~ ^[0-9]+$ ]] || die "RECOMMENDER_PORT 必须是端口数字"
  (( APP_PORT >= 1 && APP_PORT <= 65535 )) || die "RECOMMENDER_PORT 应在 1~65535 之间"
  port_is_busy "$APP_PORT" && die "指定端口 $APP_PORT 已被占用"
else
  if ! port_is_busy 5000; then
    APP_PORT=5000
  elif ! port_is_busy 5050; then
    APP_PORT=5050
    echo "[提示] 5000 已被 macOS AirPlay 等服务占用，自动改用 5050。"
  else
    die "5000 和 5050 均被占用，请先释放端口或设置 RECOMMENDER_PORT"
  fi
fi

export RECOMMENDER_HOST="$APP_HOST"
export RECOMMENDER_PORT="$APP_PORT"

echo "[1/4] 检查本地 Ollama..."
OLLAMA_BIN=""
for candidate in "$(command -v ollama 2>/dev/null)" /usr/local/bin/ollama /opt/homebrew/bin/ollama; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    OLLAMA_BIN="$candidate"
    break
  fi
done
[[ -n "$OLLAMA_BIN" ]] || die "找不到 ollama 命令"
if http_ready "$OLLAMA_URL/api/tags"; then
  echo "      Ollama 已在运行，直接复用。"
else
  "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
  OLLAMA_PID=$!
  OLLAMA_OWNED=1
  for _ in {1..60}; do
    http_ready "$OLLAMA_URL/api/tags" && break
    kill -0 "$OLLAMA_PID" >/dev/null 2>&1 || break
    sleep 0.5
  done
  if ! http_ready "$OLLAMA_URL/api/tags"; then
    echo "----- Ollama 日志 -----" >&2
    tail -n 30 "$OLLAMA_LOG" >&2 2>/dev/null || true
    die "Ollama 启动失败"
  fi
  echo "      Ollama 已启动。"
fi

if ! /usr/bin/curl -fsS --max-time 3 "$OLLAMA_URL/api/tags" \
  | grep -q '"qwen2.5vl:3b"'; then
  echo "[警告] 未找到视觉模型 qwen2.5vl:3b，视觉选品可能不可用。"
fi

echo "[2/4] 启动项目二多模态桥接（$BRIDGE_PORT）..."
if http_ready "$BRIDGE_URL/health"; then
  echo "      桥接服务已在运行，直接复用。"
elif port_is_busy "$BRIDGE_PORT"; then
  die "端口 $BRIDGE_PORT 已被其他程序占用，且不是健康的桥接服务"
else
  (
    cd "$PROJECT_ROOT/multimodal_ai_project1" || exit 1
    exec "$BRIDGE_PYTHON" -u bridge_server.py
  ) >"$BRIDGE_LOG" 2>&1 &
  BRIDGE_PID=$!
  BRIDGE_OWNED=1
  for _ in {1..120}; do
    http_ready "$BRIDGE_URL/health" && break
    kill -0 "$BRIDGE_PID" >/dev/null 2>&1 || break
    sleep 0.5
  done
  if ! http_ready "$BRIDGE_URL/health"; then
    echo "----- 桥接服务日志 -----" >&2
    tail -n 40 "$BRIDGE_LOG" >&2 2>/dev/null || true
    die "多模态桥接服务启动失败"
  fi
  echo "      桥接服务已启动。"
fi

echo "[3/4] 启动推荐服务（$APP_HOST:$APP_PORT）..."
if [[ "$APP_HOST" == "0.0.0.0" ]]; then
  BROWSER_HOST="127.0.0.1"
else
  BROWSER_HOST="$APP_HOST"
fi
APP_HOME_URL="http://${BROWSER_HOST}:${APP_PORT}/home"
APP_HEALTH_URL="http://${BROWSER_HOST}:${APP_PORT}/health"

if [[ "${RECOMMENDER_OPEN_BROWSER:-1}" != "0" ]]; then
  (
    for _ in {1..180}; do
      if http_ready "$APP_HEALTH_URL"; then
        /usr/bin/open "$APP_HOME_URL" >/dev/null 2>&1 || true
        exit 0
      fi
      sleep 0.5
    done
  ) &
  BROWSER_WAIT_PID=$!
fi

echo "[4/4] 服务就绪后将打开：$APP_HOME_URL"
echo "推荐日志：$APP_LOG"
echo "桥接日志：$BRIDGE_LOG"
echo
echo "终端保持打开时项目可用；按 Ctrl+C 停止。"
echo "----------------------------------------------"

"$APP_PYTHON" web/app.py 2>&1 | tee -a "$APP_LOG"
APP_EXIT_CODE=$?
exit "$APP_EXIT_CODE"
