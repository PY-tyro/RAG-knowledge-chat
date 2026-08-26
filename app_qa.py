"""
智能客服问答界面

基于 Streamlit 构建的聊天式 Web 应用，提供智能客服对话功能：
- 加载 RAG 服务（检索增强生成），结合知识库和 LLM 回答用户问题
- 支持流式输出，逐字显示 AI 回答内容
- 基于 session_id 持久化对话历史，刷新页面后仍可继续对话
- 使用 LangChain RunnableWithMessageHistory 管理消息记录

运行方式: streamlit run app_qa.py
"""

import streamlit as st
import uuid
from rag import RagService
from log_config import setup_logging
from security import detect_injection

setup_logging()

# 标题
st.title("智能客服")
st.divider()    # 分隔符

# 每个浏览器会话生成唯一 session_id，实现多用户对话历史隔离
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex

if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"你好,有什么可以帮助你?"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    # Prompt 注入防护：命中可疑关键词时不调用 LLM，直接拦截并友好提示
    injection_hit = detect_injection(prompt)
    if injection_hit:
        st.warning(f"检测到疑似提示词注入（命中「{injection_hit}」），已拦截。")
        safe_reply = "抱歉，我无法处理这类请求，请正常提问。"
        st.chat_message("assistant").write(safe_reply)
        st.session_state["message"].append({"role": "assistant", "content": safe_reply})
    else:
        ai_res_list = []

        def capture(generator, cache_list):
            """边流式输出边收集片段，最后拼接成完整回答存入历史"""
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        with st.spinner("AI思考中..."):
            try:
                session_config = {"configurable": {"session_id": st.session_state["session_id"]}}
                res_stream = st.session_state["rag"].chain.stream({"input": prompt}, session_config)
                st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
            except Exception as e:
                # LLM 调用失败（Key 无效 / 网络 / 余额不足等）时，友好提示而不是抛 traceback
                st.error(f"AI 回答失败：{e}")
            else:
                # 仅在生成成功时才把回答写入历史，避免保存残缺/错误内容
                st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})
                # 展示本次回答检索到的知识库来源（文件名），增强回答可信度
                try:
                    sources = st.session_state["rag"].get_sources(prompt)
                except Exception:
                    sources = []
                if sources:
                    st.caption("参考来源：" + "、".join(sources))

# list = ["a","b","c"]     "".join(list) ->abc
# list = ["a","b","c"]     ",".join(list) ->a,b,c
