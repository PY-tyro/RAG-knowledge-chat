"""
项目集中配置文件

集中管理整个项目的可调参数，方便统一修改和维护，主要包括：
- 文件路径：MD5 记录文件路径、Chroma 向量数据库持久化目录
- 文本分割参数：chunk_size（块大小）、chunk_overlap（重叠量）、分隔符列表
- 检索参数：similarity_threshold（返回文档数量 k 值）
- 模型名称：向量化模型（text-embedding-v4）、对话模型（qwen3-max）
- 会话配置：session_id 等 LangChain 可配置参数

所有模块通过 import config_data as config 引用此文件的配置项。
"""

md5_path = "./md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"
# 集合索引配置：使用余弦距离（cosine）。入库和检索必须用同一配置，否则距离/阈值会错乱
chroma_collection_configuration = {"hnsw": {"space": "cosine"}}

# spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n",".","!","?",","," ","","！","？","。","，","",]

max_split_char_number = 1000        # 文本分割的阈值

top_k = 4        # 检索返回匹配的文档数量（Top-K 值，越大答案越完整但耗时越长）

# 相似度分数阈值：只召回余弦相似度 >= 该值的片段，防止无关内容干扰 LLM 回答
# 值越高越精准（可能漏召回），越低越全（可能混入无关内容）
similarity_score_threshold = 0.5

embedding_model_name = "text-embedding-v4"

# 入库元数据中的操作者标识（不要硬编码个人信息，统一通过配置维护；后续多用户时改为真实用户 ID）
operator = "admin"

chat_model_name = "qwen3-max"

# 注意：session_id 不再硬编码，而是在 app_qa.py 中按浏览器会话动态生成（uuid），
# 使每个用户拥有独立的对话历史文件（文件名即 session_id），避免多用户互相串扰。