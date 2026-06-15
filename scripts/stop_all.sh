#!/usr/bin/env bash
# 停止所有 Agent 和 Web 服务

patterns=(
    "planning_agent/main.py"
    "investment_agent/main.py"
    "statistics_agent/main.py"
    "main_agent/server.py"
    "streamlit run web/app.py"
)

for pattern in "${patterns[@]}"; do
    pids=$(pgrep -f "$pattern" || true)
    if [ -n "$pids" ]; then
        echo "[停止] $pattern -> $pids"
        kill $pids 2>/dev/null || true
    fi
done

echo "完成"
