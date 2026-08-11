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

# spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n",".","!","?",","," ","","！","？","。","，","",]

max_split_char_number = 1000        # 文本分割的阈值

similarity_threshold = 1        # 检索返回匹配的文档数量

embedding_model_name = "text-embedding-v4"

chat_model_name = "qwen3-max"

session_config = {
        "configurable":{
            'session_id':"user_001",
        }
    }