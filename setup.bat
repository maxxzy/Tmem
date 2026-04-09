@echo off
REM ============================================================
REM TMem 环境准备脚本 (Windows)
REM ============================================================

echo ============================
echo  TMem 环境准备
echo ============================

REM 1. 安装 Python 依赖
echo.
echo [1/3] 安装 Python 依赖...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Python 依赖安装失败
    pause
    exit /b 1
)
echo [OK] Python 依赖安装完成

REM 2. 检查 Neo4j
echo.
echo [2/3] 检查 Neo4j...
echo   Neo4j 需要手动安装和启动:
echo     方式A: Docker:  docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/tmem2024 neo4j:latest
echo     方式B: 桌面版:  https://neo4j.com/download/
echo     方式C: 跳过:    测试时加 --no-neo4j 参数即可
echo.

REM 3. 检查 Qdrant
echo [3/3] 检查 Qdrant...
echo   Qdrant 需要手动安装和启动:
echo     方式A: Docker:  docker run -d -p 6333:6333 qdrant/qdrant
echo     方式B: 二进制:  https://qdrant.tech/documentation/quick-start/
echo     方式C: 跳过:    测试时加 --no-qdrant 参数即可
echo.

echo ============================
echo  环境准备完成!
echo ============================
echo.
echo 运行测试:
echo   python run_eval.py --mode quick_test --no-neo4j --no-qdrant
echo.
pause
