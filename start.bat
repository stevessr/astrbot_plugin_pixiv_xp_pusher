@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: 解析命令行参数
set MODE=launcher
set SKIP_WIZARD=0
:parse_args
if "%~1"=="" goto :run
if /i "%~1"=="--once" set MODE=once
if /i "%~1"=="-o" set MODE=once
if /i "%~1"=="--skip" set SKIP_WIZARD=1
if /i "%~1"=="-s" set SKIP_WIZARD=1
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help
shift
goto :parse_args

:help
echo Pixiv-XP-Pusher 启动脚本
echo.
echo 用法: start.bat [选项]
echo.
echo 选项:
echo   (无参数)      启动引导程序/主菜单
echo   --once, -o    立即执行一次推送
echo   --skip, -s    跳过首次运行向导
echo   --help, -h    显示此帮助信息
echo.
goto :eof

:run
:: 尝试激活 Conda
where conda >nul 2>&1
if not errorlevel 1 (
    call conda activate pixiv-xp >nul 2>&1
)

:: 跳过向导：创建标记文件
if "%SKIP_WIZARD%"=="1" (
    if not exist ".initialized" (
        echo done> .initialized
        echo [已跳过首次运行向导]
    )
)

:: 运行
if "%MODE%"=="once" (
    echo [立即执行模式]
    python main.py --once
) else (
    python launcher.py
)
if errorlevel 1 pause
