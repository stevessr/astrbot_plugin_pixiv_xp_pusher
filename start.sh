#!/bin/bash
# Pixiv-XP-Pusher 本地启动脚本 (对应 start.bat)

set -e
cd "$(dirname "$0")"

# 默认参数
MODE="launcher"
SKIP_WIZARD=0

# 帮助信息
show_help() {
    echo "Pixiv-XP-Pusher 启动脚本"
    echo ""
    echo "用法: ./start.sh [选项]"
    echo ""
    echo "选项:"
    echo "  (无参数)      启动引导程序/主菜单"
    echo "  --once, -o    立即执行一次推送"
    echo "  --skip, -s    跳过首次运行向导"
    echo "  --help, -h    显示此帮助信息"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --once|-o)
            MODE="once"
            shift
            ;;
        --skip|-s)
            SKIP_WIZARD=1
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 尝试激活 Conda 环境
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate pixiv-xp 2>/dev/null || true
fi

# 跳过向导：创建标记文件
if [[ "$SKIP_WIZARD" == "1" && ! -f ".initialized" ]]; then
    echo "done" > .initialized
    echo "[已跳过首次运行向导]"
fi

# 运行
if [[ "$MODE" == "once" ]]; then
    echo "[立即执行模式]"
    python main.py --once
else
    python launcher.py
fi
