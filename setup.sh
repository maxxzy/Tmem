#!/bin/bash
# ============================================================
# TMem 环境准备脚本 (Linux/Mac)
# ============================================================

set -e
echo "============================"
echo " TMem 环境准备"
echo "============================"

# 1. 安装 Python 依赖
echo ""
echo "[1/3] 安装 Python 依赖..."
pip install -r requirements.txt
echo "[OK] Python 依赖安装完成"

# 2. 启动 Neo4j (Docker)
echo ""
echo "[2/3] Neo4j..."
if command -v docker &> /dev/null; then
    if docker ps --format '{{.Names}}' | grep -q tmem-neo4j; then
        echo "  Neo4j 已运行"
    else
        echo "  启动 Neo4j 容器..."
        docker run -d --name tmem-neo4j \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/tmem2024 \
            neo4j:latest
        echo "  等待 Neo4j 启动..."
        sleep 10
    fi
else
    echo "  [WARN] Docker 未安装, 测试时请加 --no-neo4j"
fi

# 3. 启动 Qdrant (Docker)
echo ""
echo "[3/3] Qdrant..."
if command -v docker &> /dev/null; then
    if docker ps --format '{{.Names}}' | grep -q tmem-qdrant; then
        echo "  Qdrant 已运行"
    else
        echo "  启动 Qdrant 容器..."
        docker run -d --name tmem-qdrant \
            -p 6333:6333 \
            qdrant/qdrant
        sleep 5
    fi
else
    echo "  [WARN] Docker 未安装, 测试时请加 --no-qdrant"
fi

echo ""
echo "============================"
echo " 环境准备完成!"
echo "============================"
echo ""
echo "运行测试:"
echo "  python run_eval.py --mode quick_test --no-neo4j --no-qdrant"
