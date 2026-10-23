#!/bin/zsh

# 一键关闭推荐系统与多模态桥接。
#
# 只终止本项目目录下运行的服务，不按端口误杀 AirPlay 等系统进程，也不主动
# 结束 Ollama：如果 Ollama 是「启动项目.command」启动的，一键启动脚本自己的
# 清理逻辑会关闭它；如果它原本就在运行，则继续供其他程序共享使用。

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
MULTIMODAL_ROOT="$PROJECT_ROOT/multimodal_ai_project1"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ -n "${1:-}" ]]; then
  echo "用法：关闭项目.command [--dry-run]" >&2
  exit 2
fi

print_header() {
  echo "=============================================="
  echo " 电商商品智能推荐系统 · 一键关闭"
  echo "=============================================="
  echo "项目目录：$PROJECT_ROOT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "模式：安全检查（只显示，不终止进程）"
  fi
  echo
}

cwd_of_pid() {
  local pid="$1"
  /usr/sbin/lsof -a -p "$pid" -d cwd -Fn 2>/dev/null \
    | /usr/bin/sed -n 's/^n//p' \
    | /usr/bin/head -n 1
}

find_project_pids() {
  local marker="$1"
  local expected_cwd="$2"
  local line pid command cwd

  /bin/ps -axo pid=,command= \
    | while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"
        pid="${line%% *}"
        command="${line#* }"
        [[ "$command" == *"$marker"* ]] || continue
        cwd="$(cwd_of_pid "$pid")"
        [[ "$cwd" == "$expected_cwd" ]] || continue
        echo "$pid"
      done \
    | /usr/bin/sort -n -u
}

stop_one() {
  local pid="$1"
  local label="$2"
  local attempt

  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[将关闭] $label（PID $pid）"
    return 0
  fi

  echo "[关闭中] $label（PID $pid）"
  kill -TERM "$pid" >/dev/null 2>&1 || true
  for attempt in {1..50}; do
    kill -0 "$pid" >/dev/null 2>&1 || return 0
    sleep 0.1
  done

  echo "[强制结束] $label 未在 5 秒内退出（PID $pid）"
  kill -KILL "$pid" >/dev/null 2>&1 || true
  for attempt in {1..20}; do
    kill -0 "$pid" >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  echo "[失败] 无法结束 $label（PID $pid）" >&2
  return 1
}

stop_service() {
  local label="$1"
  local marker="$2"
  local expected_cwd="$3"
  local found=0
  local failed=0
  local pid
  local -a pids

  pids=("${(@f)$(find_project_pids "$marker" "$expected_cwd")}")
  pids=("${(@)pids:#}")

  for pid in "${pids[@]}"; do
    found=1
    stop_one "$pid" "$label" || failed=1
  done

  if [[ "$found" -eq 0 ]]; then
    echo "[未运行] $label"
  fi
  return "$failed"
}

wait_for_launcher_cleanup() {
  # 子服务退出后，一键启动脚本会执行 EXIT 清理，并关闭它自己启动的 Ollama。
  # 这里留出短暂时间，不主动查找或结束任何 ollama 进程。
  [[ "$DRY_RUN" -eq 1 ]] && return 0
  sleep 0.5
}

failed=0
print_header

# 先关闭 Web 服务，再关闭桥接。视觉模型随后会释放显存/内存。
stop_service \
  "推荐服务（web/app.py）" \
  "web/app.py" \
  "$PROJECT_ROOT" \
  || failed=1

stop_service \
  "多模态桥接（bridge_server.py）" \
  "bridge_server.py" \
  "$MULTIMODAL_ROOT" \
  || failed=1

wait_for_launcher_cleanup

echo
if [[ "$failed" -ne 0 ]]; then
  echo "[结果] 部分服务未能关闭，请检查上面的错误信息。"
  exit_code=1
elif [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[结果] 安全检查完成，没有终止任何进程。"
  exit_code=0
else
  echo "[结果] 项目服务已关闭。"
  echo "       Ollama 若原本就在运行，会继续保留供其他程序使用。"
  exit_code=0
fi

# 双击 .command 运行时保留窗口，方便查看结果；命令行调用不在终端中等待。
if [[ "$DRY_RUN" -eq 0 && -t 0 ]]; then
  echo
  read -r "?按回车键关闭窗口..." _
fi

exit "$exit_code"
