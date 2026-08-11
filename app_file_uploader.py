"""
知识库文件上传界面

基于 Streamlit 构建的知识库管理 Web 应用，提供文档上传入库功能：
- 支持上传 TXT 文本文件，读取内容后调用 KnowledgeBaseService 进行向量化入库
- 展示上传文件的基本信息（文件名、格式、大小）
- 入库前通过 MD5 去重检测，避免重复处理相同内容
- Streamlit 特性：页面元素变化时，代码会重新执行一遍

运行方式: streamlit run app_file_uploader.py
"""

import streamlit as st
from knowledge_base import KnowledgeBaseService
import time

# 添加网页标题
st.title("知识库更新服务")

# 添加文件上传服务 -> file_uploader
uploader_file = st.file_uploader(
    label= "请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False     # false表示仅接受一个文件的上传
)


# st.session_state是一个字典

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploader_file is not None:
    # 提取文件信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024 # kb

    st.subheader(f"文件名: {file_name}")
    st.write(f"格式: {file_type} | 大小: {file_size:.2f} KB")

    # 获取文件里的内容 -> getvalue ->bytes ->decode('utf-8')
    text = uploader_file.getvalue().decode("utf-8")


    with st.spinner("载入知识库中。。。"):      # 在spinner内的代码执行过程中,会有一个转圈动画
        time.sleep(1)       # 必须导入time模块
        result = st.session_state["service"].upload_by_str(text,file_name)
        st.write(result)
    


