"""
RAG（检索增强生成）服务模块

本模块是智能客服的核心，封装了完整的 RAG 流程：
- 向量检索：通过 VectorStoreService 从 Chroma 中召回与用户问题最相关的文档片段
- Prompt 构建：使用 ChatPromptTemplate 将系统提示、参考资料、对话历史和用户问题组合
- LLM 调用：通过阿里云通义千问（qwen3-max）生成专业回答
- 对话记忆：使用 RunnableWithMessageHistory 结合 FileChatMessageHistory 实现多轮对话
- 链式编排：基于 LangChain LCEL（LangChain Expression Language）构建处理链
  Pipeline: 用户输入 → 向量检索 → 格式化文档 → 拼接 Prompt → LLM 生成 → 字符串输出

运行方式: python rag.py（内置测试用例，直接调用 invoke 测试）
"""

from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
load_dotenv()
import config_data as config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from file_history_store import get_history


def print_prompt(prompt):

    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt

class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate(
            [
                ("system","以我提供的已知参考资料为主,"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                 ("system","并且我提供用户的对话历史记录,如下: "),
                 MessagesPlaceholder("history"),
                 ("user","请回答用户问题:{input}")
            ]
        )

        self.chat_model = ChatTongyi(
            model=config.chat_model_name
        )

        self.chain = self.__get_chain()


    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n文档原数据:{doc.metadata}\n\n"
            return formatted_str

        def format_for_retriever(value:dict) -> str:
            
            return value["input"]

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) |self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
            
        )
        return conversation_chain

if __name__ == '__main__':
    # session id 配置
    session_config = {
        "configurable":{
            'session_id':"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"针织毛衣如何保养?"},session_config )
    print(res)