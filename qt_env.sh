# Qt环境设置脚本
# 清理Qt相关环境变量
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

# 设置基本Qt环境
export QT_QPA_PLATFORM=xcb
export QT_LOGGING_RULES="*.debug=false"
export QT_X11_NO_MITSHM=1
export QT_XCB_GL_INTEGRATION=""

# 确保显示设置
if [[ -z "$DISPLAY" ]]; then
    export DISPLAY=:0
fi

echo "Qt环境已设置"
echo "DISPLAY: $DISPLAY"
echo "QT_QPA_PLATFORM: $QT_QPA_PLATFORM"
