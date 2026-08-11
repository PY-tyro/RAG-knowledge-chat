import streamlit as st
import time
from rag import RagService
import config_data as config

# 标题
st.title("智能客服")
st.divider()    # 分隔符

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

    ai_res_list = []
    with st.spinner("AI思考中..."):
        # 直接输出
        # res = st.session_state["rag"].chain.invoke({"input":prompt},config.session_config)
        # st.chat_message("assistant").write(res)
        # st.session_state["message"].append({"role":"assistant","content":res})

        res_stream = st.session_state["rag"].chain.stream({"input":prompt},config.session_config)
        # yield 迭代器

        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        # st.chat_message("assistant").write(res_stream)
        # st.session_state["message"].append({"role":"assistant","content":res_stream}) # content 要的是字符串stream输出的不符合条件,存储会失败 而且这个流已经输出了,里面没有内容了
        
        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role":"assistant","content":"".join(ai_res_list)})

# list = ["a","b","c"]     "".join(list) ->abc
# list = ["a","b","c"]     ",".join(list) ->a,b,c
