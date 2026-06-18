#!/usr/bin/env bash
# 启动三个业务 Agent、主控 Agent 和 Web 聊天页

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

start_service() {
    local name="$1"
    local cmd="$2"
    local log_file="$LOG_DIR/${name}.log"

    if pgrep -f "$cmd" >/dev/null 2>&1; then
        echo "[跳过] $name 已在运行"
        return
    fi

    nohup bash -c "$cmd" >"$log_file" 2>&1 &
    echo "[启动] $name (日志: $log_file)"
}

echo "=== 启动业务 Agent ==="
start_service "planning" "python -u planning_agent/main.py"
start_service "investment" "python -u investment_agent/main.py"
start_service "statistics" "python -u statistics_agent/main.py"

echo "等待业务 Agent 就绪..."
sleep 4

echo "=== 启动主控 Agent ==="
start_service "main" "python -u main_agent/server.py"
sleep 3

echo "=== 启动 Web 聊天页 ==="
start_service "web" "uvicorn web.server:app --host 0.0.0.0 --port 8501"
sleep 3

echo ""
echo "=== 服务地址 ==="
echo "规划 Agent:   http://localhost:8001"
echo "投资 Agent:   http://localhost:8002"
echo "统计 Agent:   http://localhost:8003"
echo "主控 Agent:   http://localhost:8000"
echo "Web 聊天页:   http://localhost:8501"
echo ""
echo "查看进程: pgrep -fl 'planning_agent/main|investment_agent/main|statistics_agent/main|main_agent/server|uvicorn web.server'"
echo "停止服务: bash scripts/stop_all.sh"
