"""
对话历史本地文件存储模块

基于 LangChain BaseChatMessageHistory 实现的自定义对话历史持久化方案：
- 将对话消息以 JSON 格式存储在本地文件中，一个 session_id 对应一个文件
- 支持消息的添加（add_messages）、读取（messages）和清空（clear）
- 使用 LangChain 官方的 message_to_dict / messages_from_dict 进行序列化和反序列化
- 配合 LangChain RunnableWithMessageHistory 使用，实现对话记忆功能

存储路径: ./chat_history/<session_id>
"""

from langchain_core.chat_history import BaseChatMessageHistory
import os,json
from typing import Sequence
from langchain_core.messages import BaseMessage,message_to_dict,messages_from_dict

def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id   # 会话id
        self.storage_path = storage_path  # 不同会话id的存贮文件,所在的文件路径
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path,self.session_id)

        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_messages(self,messages:Sequence[BaseMessage]) -> None:
            # Sequence序列 类似list,tuple
            all_messages = list(self.messages)  # 已有的消息列表
            all_messages.extend(messages)   # 新的和已有的融合成一个list

            # 将新的数据同步写入到本地文件中
            # 类对象写入文件 -> 一堆二进制
            # 为了方便,可以将BaseMessage消息转为字典(借助json模块以json字符串写入文件)
            # 官方message_to_dict:单个消息对象(BaseMessage类实例) -> 字典
            new_messages = []
            # for message in all_messages:
            #     d = message_to_dict(message)
            #     new_messages.append(d)

            new_messages = [message_to_dict(message) for message in all_messages]  # (列表推导式) 与上面代码作用等同
            # 将数据写入文件
            with open(self.file_path,"w",encoding="utf-8") as f:
                json.dump(new_messages,f)

    @property   # @property装饰器将messages方法变成成员属性用
    def messages(self) ->list[BaseMessage]:
        #  当前文件内: list[字典]
        try:
             with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
             return []
    def clear(self) -> None:
         with open(self.file_path,"w",encoding="utf-8") as f:
              json.dump([],f)