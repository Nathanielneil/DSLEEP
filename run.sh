# #!/bin/bash
# # 疲劳检测系统启动脚本

# set -e

# # 颜色定义
# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m' # No Color

# echo -e "${BLUE}=================================================="
# echo -e "疲劳检测系统 - Ubuntu版启动脚本"
# echo -e "==================================================${NC}"

# # 检查是否在dsleep环境中
# if [[ "$CONDA_DEFAULT_ENV" != "dsleep" ]]; then
#     echo -e "${RED}错误：请先激活dsleep conda环境${NC}"
#     echo -e "${YELLOW}运行: conda activate dsleep${NC}"
#     exit 1
# fi

# # 显示环境信息
# echo -e "${GREEN}当前环境信息:${NC}"
# echo "  Conda环境: $CONDA_DEFAULT_ENV"
# echo "  Python路径: $(which python)"
# echo "  工作目录: $(pwd)"

# # 检查显示环境
# if [[ -n "$DISPLAY" ]]; then
#     echo -e "  显示: ${GREEN}$DISPLAY${NC}"
# else
#     echo -e "  显示: ${YELLOW}未设置 (GUI可能无法工作)${NC}"
# fi

# # 检查摄像头设备
# if ls /dev/video* >/dev/null 2>&1; then
#     echo -e "  摄像头: ${GREEN}检测到设备${NC}"
# else
#     echo -e "  摄像头: ${RED}未检测到设备${NC}"
# fi

# echo ""

# # 解析命令行参数
# MODE="auto"
# SKIP_TEST=false

# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --gui)
#             MODE="gui"
#             shift
#             ;;
#         --console)
#             MODE="console"
#             shift
#             ;;
#         --test)
#             MODE="test"
#             shift
#             ;;
#         --skip-test)
#             SKIP_TEST=true
#             shift
#             ;;
#         --help|-h)
#             echo "用法: $0 [选项]"
#             echo ""
#             echo "选项:"
#             echo "  --gui        强制使用GUI模式"
#             echo "  --console    强制使用命令行模式"
#             echo "  --test       仅运行测试"
#             echo "  --skip-test  跳过测试直接启动"
#             echo "  --help, -h   显示此帮助信息"
#             echo ""
#             echo "示例:"
#             echo "  $0           # 自动选择模式"
#             echo "  $0 --gui     # GUI模式"
#             echo "  $0 --console # 命令行模式"
#             exit 0
#             ;;
#         *)
#             echo -e "${RED}未知参数: $1${NC}"
#             echo "使用 --help 查看帮助"
#             exit 1
#             ;;
#     esac
# done

# # 运行测试（除非跳过）
# if [[ "$SKIP_TEST" != "true" && "$MODE" != "test" ]]; then
#     echo -e "${BLUE}运行快速测试...${NC}"
#     if python test_system.py >/dev/null 2>&1; then
#         echo -e "${GREEN}✓ 系统测试通过${NC}"
#     else
#         echo -e "${YELLOW}  系统测试未完全通过，但将继续启动${NC}"
#         echo -e "${YELLOW}   如需详细测试信息，运行: python test_system.py${NC}"
#     fi
#     echo ""
# fi

# # 根据模式启动程序
# case $MODE in
#     "test")
#         echo -e "${BLUE}运行完整系统测试...${NC}"
#         python test_system.py
#         ;;
#     "gui")
#         echo -e "${BLUE}启动GUI模式...${NC}"
#         python main.py
#         ;;
#     "console")
#         echo -e "${BLUE}启动命令行模式...${NC}"
#         python main.py --console
#         ;;
#     "auto")
#         # 自动选择模式
#         if [[ -n "$DISPLAY" ]] && xset q >/dev/null 2>&1; then
#             echo -e "${BLUE}自动选择GUI模式...${NC}"
#             python main.py
#         else
#             echo -e "${BLUE}显示环境不可用，使用命令行模式...${NC}"
#             python main.py --console
#         fi
#         ;;
# esac

# echo -e "${GREEN}程序已退出${NC}"