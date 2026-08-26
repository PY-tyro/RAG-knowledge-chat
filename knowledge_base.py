"""
知识库管理服务模块

提供文档向量化入库的核心功能，主要包含：
- KnowledgeBaseService 类：封装 Chroma 向量库和文本分割器的初始化与操作
- upload_by_str()：将文本字符串向量化后存入 Chroma 数据库，支持元数据记录
- MD5 去重机制：通过 check_md5() 和 save_md5() 防止相同内容重复入库
- 文本分割：使用 RecursiveCharacterTextSplitter 按分隔符递归分割长文本
- 向量化：通过阿里云 DashScope Embeddings（text-embedding-v4）生成文本向量

MD5 记录文件: ./md5.text
向量数据库目录: ./chroma_db/
"""

import os
import logging
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

def check_md5(md5_str:str):
    """检查传入的md5字符串是否已经被处理过了
        return false(md5未处理)     true(已经处理过了,已有记录)
    """
    if not os.path.exists(config.md5_path):
        # if 进入表示文件不存在,那肯定没有处理过这个md5
        open(config.md5_path,"w",encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path,"r",encoding="utf-8").readlines():
            line = line.strip()     # 处理字符串前后的空格和回车
            if line == md5_str:
                return True     # 已处理过
        return False


def save_md5(md5_str:str):
    """将传入的md5字符串记录到文件内保存"""
    with open(config.md5_path,"a",encoding="utf-8") as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str: str,encoding="utf-8"):
    """将传入的字符串转换为md5字符串"""


    # 将字符串转化为bytes字节数据
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5对象
    md5_obj = hashlib.md5()     # 得到md5对象
    md5_obj.update(str_bytes)   # 更新内容(传入即将要转换的字节数组)
    md5_hex = md5_obj.hexdigest() # 得到md5的十六字符字符串

    return md5_hex



class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在则创建,如果存在则跳过
        os.makedirs(config.persist_directory,exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,  # 数据库表名
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory, # 数据库本地存储文件夹
            collection_configuration=config.chroma_collection_configuration,  # 与 vector_stores.py 用同一余弦配置
        )  # 向量存储的实例Chroma向量库对象


        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,     # 分割后的文本最大长度
            chunk_overlap = config.chunk_overlap, # 连续文本段之间的字符重叠数量
            separators=config.separators,       # 自然段落划分的符号
            length_function = len,          # 使用PYthon自带的len函数做长度统计的依据
        )     # 文本分割器的对象

    def upload_by_str(self,data:str,filename):
        """将传入的字符串,进行向量化,存入向量数据库中"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > config.max_split_char_number:
            knowledge_chunks:list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":config.operator
        }



        try:
            # 更新场景：若该文件名之前已入库（内容有改动），先删除旧片段，避免旧数据残留
            self.chroma.delete(where={"source": filename})
            self.chroma.add_texts(    # 内容加载到向量库中(会调用 embedding API，可能因 Key 无效/网络/余额不足失败)
                # iterable -> list \ tuple
                knowledge_chunks,
                metadatas = [metadata for _ in knowledge_chunks]
            )
        except Exception as e:   # 捕获 embedding/网络/API 异常，返回友好提示而不是让页面崩溃
            logger.exception("入库失败: %s", filename)
            return f"[失败]内容载入向量库失败：{e}"

        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库"

    def list_documents(self):
        """列出知识库中已有的文档（按 source 去重，返回 {文件名: 片段数量}）"""
        result = self.chroma.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        sources = {}
        for meta in metadatas:
            src = meta.get("source", "未知")
            sources[src] = sources.get(src, 0) + 1
        return sources

    def delete_by_source(self, source: str):
        """按文件名删除知识库中的文档片段"""
        try:
            self.chroma.delete(where={"source": source})
            return f"[成功]已删除文档：{source}"
        except Exception as e:
            logger.exception("删除失败: %s", source)
            return f"[失败]删除文档失败：{e}"

if __name__ == '__main__':
    service = KnowledgeBaseService()
    res = service.upload_by_str("周杰伦","testfile")
    print(res)
