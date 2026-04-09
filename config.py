"""
TMem 系统全局配置
包含主题感知记忆系统的所有可调参数
"""

import os

# ======================== 嵌入模型配置 ========================
# 嵌入后端: "sentence-transformers" (本地加载) 或 "ollama" (通过 Ollama API)
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "ollama")
# sentence-transformers 后端使用的模型名（HuggingFace 模型 ID 或本地路径）
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Ollama 后端使用的模型名
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "mahonzhan/all-MiniLM-L6-v2")
# Ollama embedding API 地址
OLLAMA_EMBEDDING_URL = os.environ.get("OLLAMA_EMBEDDING_URL", "http://127.0.0.1:11434/api/embed")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 的输出维度

# ======================== LLM 配置 ========================
# 默认使用 OpenAI API；若使用 Ollama 本地模型，设置环境变量：
#   export LLM_MODEL="qwen2.5:7b"                       # Ollama 中的模型名
#   export LLM_BASE_URL="http://<server_ip>:11434/v1"    # Ollama 的 OpenAI 兼容端点
#   export LLM_API_KEY="ollama"                          # Ollama 不校验 key，任意非空值即可
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:30b")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "ollama"))
LLM_TEMPERATURE = 0.3  # 较低温度以获得更稳定的结构化输出

# ======================== Neo4j 图数据库配置 ========================
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:17687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "MyPass123")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# ======================== Qdrant 向量数据库配置 ========================
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "16333"))
QDRANT_COLLECTION_MEMORIES = "tmem_memories"      # 记忆向量集合
QDRANT_COLLECTION_TOPICS = "tmem_topics"          # 主题向量集合（标签+摘要嵌入）

# ======================== LoCoMo 数据集配置 ========================
LOCOMO_DATA_PATH = os.path.join(os.path.dirname(__file__), "locomo", "data", "locomo10.json")

# ======================== 主题分段配置 ========================
# 相邻对话语义相似度低于此阈值时，视为候选主题边界
TOPIC_BOUNDARY_SIMILARITY_THRESHOLD = 0.5
# 转折词列表，出现这些词时辅助判断主题边界
TRANSITION_WORDS = [
    "By the way", "Speaking of", "On another note",
    "Also", "However", "Anyway", "Actually",
    "So", "Well", "Oh", "Let me change the subject",
]
# 新主题标签与已有主题匹配的相似度阈值，超过则归入已有主题
TOPIC_MERGE_SIMILARITY_THRESHOLD = 0.75

# ======================== 主题 DAG 配置 ========================
# 主题综合相似度的权重系数 (标签嵌入, 摘要嵌入, 关键词Jaccard)
TOPIC_SIM_WEIGHTS = (0.4, 0.3, 0.3)  # alpha1, alpha2, alpha3

# HAC 层次凝聚聚类的多粒度距离阈值（从粗到细）
HAC_DISTANCE_THRESHOLDS = [0.7, 0.5, 0.3]

# 多归属阈值：主题与非当前父节点的上层节点相似度超过此值时添加多父边
MULTI_PARENT_THRESHOLD = 0.6  # beta

# 主题摘要更新触发条件：每新增这么多条记忆就重新生成摘要
SUMMARY_UPDATE_INTERVAL = 5

# 主题漂移检测的 JSD 阈值
TOPIC_DRIFT_JSD_THRESHOLD = 0.3  # delta

# 主题合并的相似度阈值
TOPIC_MERGE_THRESHOLD = 0.85  # mu

# 触发主题合并时的最低记忆数量上限（两个主题的记忆数量都需低于该值）
TOPIC_MERGE_MIN_MEMORY_COUNT = 10

# ======================== 主题关联图配置 ========================
# 共现建边条件
NPMI_THRESHOLD = 0.1  # tau_co，NPMI 超过此阈值才建边
MIN_COOCCURRENCE = 2  # n_min，最少共现次数

# LLM 因果判断的候选对筛选区间（向量相似度）
LLM_CAUSAL_SIM_LOW = 0.3
LLM_CAUSAL_SIM_HIGH = 0.6

# 边权重的来源可信度权重 (共现NPMI, LLM评分, 时序NPMI)
EDGE_WEIGHT_SOURCES = (0.5, 0.3, 0.2)  # w1, w2, w3

# 边方向判定的对称性阈值
DIRECTION_EPSILON = 0.05

# 图剪枝参数
EDGE_WEIGHT_PRUNE_THRESHOLD = 0.15  # tau_w
MAX_OUT_DEGREE = 5  # 每个节点最多保留的出边数量 K

# 半衰期（天），衰减到一半权重所需的时间
HALF_LIFE_DAYS = 30  # t_half

# 衰减-强化更新的学习率
DECAY_REINFORCE_LR = 0.2  # eta
WEIGHT_MAX = 1.0  # w_max

# ======================== PPR (Personalized PageRank) 配置 ========================
PPR_DAMPING = 0.85  # alpha，阻尼系数
PPR_MAX_ITER = 20  # 最大迭代次数
PPR_CONVERGENCE_EPSILON = 1e-6  # 收敛判定阈值
PPR_TOP_K = 5  # 跨主题扩展取 top-K 个主题

# ======================== 检索配置 ========================
# 主题路由时，query 匹配主题的最大数量上限
MAX_ROUTED_TOPICS = 5
# 主题路由时，query 匹配主题的最小数量下限（不足时沿 DAG 向上扩展）
MIN_ROUTED_TOPICS = 1
# 主题内记忆检索的 top-K
INTRA_TOPIC_TOP_K = 10
# 跨主题扩展的记忆权重折扣因子
CROSS_TOPIC_WEIGHT_DISCOUNT = 0.7
# 推理关键词，出现时触发跨主题扩展
REASONING_KEYWORDS = [
    "why", "reason", "cause", "how come",
    "what caused", "what led to", "because",
    "explain", "how did",
]
# 检索得分过低阈值，低于此值触发跨主题扩展
LOW_SCORE_THRESHOLD = 0.3

# ======================== 定期重构配置 ========================
# 每新增多少条记忆后触发全局重构
GLOBAL_REBUILD_INTERVAL = 50
