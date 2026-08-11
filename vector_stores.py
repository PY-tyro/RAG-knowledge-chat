"""
向量存储服务模块

封装 Chroma 向量数据库的初始化与检索操作，是 RAG 流程的检索基础：
- VectorStoreService 类：接收 Embedding 实例，初始化 Chroma 向量存储
- get_retriever()：返回 LangChain 标准的检索器对象，可直接接入 Chain
- 检索器配置了 k 值（每次返回的文档数量），通过 config_data.similarity_threshold 控制
- 与 knowledge_base.py 共用同一个 Chroma 持久化目录和集合名

使用示例:
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()
    docs = retriever.invoke("查询内容")
"""

from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )

    def get_retriever(self):
        """返回向量检索器,方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs = {"k":config.similarity_threshold})

if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    from dotenv import load_dotenv
    load_dotenv()
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体重180斤,尺码推荐")
    print(res)

