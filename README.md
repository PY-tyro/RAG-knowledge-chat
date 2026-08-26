# 智能客服 — 基于 RAG 的知识库问答系统

## 项目简介

本项目是一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 架构的智能客服问答系统。系统预先将知识文档向量化存入 Chroma 向量数据库，当用户提问时，先从知识库中检索相关文档片段，再结合对话历史与大语言模型（LLM）生成精准、专业的回答。

项目包含两个独立的 Streamlit Web 应用：

| 应用 | 入口文件 | 功能 |
|------|----------|------|
| 智能客服 | `app_qa.py` | 提供聊天界面，用户输入问题，系统基于知识库检索并回答 |
| 知识库更新 | `app_file_uploader.py` | 上传 / 查看 / 删除 TXT 文档，自动向量化并存入知识库（支持 MD5 去重与内容更新） |

当前知识库示例内容为服装销售领域的常见问题（尺码推荐、洗涤养护、颜色选择等），可根据实际需求替换为任意领域的知识文档。

### 工作原理

1. **文档入库**：通过 `app_file_uploader.py` 上传 TXT 文件 → 文本分割（RecursiveCharacterTextSplitter）→ 调用阿里云 DashScope Embedding 生成向量 → 存入 Chroma 向量数据库（同时记录 MD5 防止重复入库）。
2. **智能问答**：用户在 `app_qa.py` 中提问 → 向量检索器从 Chroma 中召回最相关的文档片段 → 将文档片段、对话历史与用户问题拼接为 Prompt → 发送给阿里云通义千问（qwen3-max）生成回答 → 流式输出到页面，回答中标注引用编号（【资料N】），并在答案下方列出参考来源文件。
3. **对话记忆**：基于 LangChain `RunnableWithMessageHistory`，对话历史以 JSON 文件形式持久化到本地 `chat_history/` 目录。

## 技术栈

| 技术 | 用途 |
|------|------|
| [Streamlit](https://streamlit.io/) | Web 界面框架，提供聊天 UI 和文件上传交互 |
| [LangChain](https://www.langchain.com/) | RAG 流程编排（Prompt 模板、Chain、消息历史管理） |
| [Chroma](https://www.trychroma.com/) | 本地向量数据库，存储文档的向量化表示 |
| [DashScope Embeddings](https://help.aliyun.com/document_detail/2712175.html) | 阿里云文本向量模型（`text-embedding-v4`），将文本转为向量 |
| [ChatTongyi（通义千问）](https://help.aliyun.com/document_detail/2712155.html) | 阿里云大语言模型（`qwen3-max`），用于生成回答 |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | 从 `.env` 文件加载阿里云 API Key 等环境变量 |

## 项目结构

```
项目1/
├── app_qa.py              # 智能客服聊天界面（Streamlit）
├── app_file_uploader.py   # 知识库文件上传界面（Streamlit）
├── rag.py                 # RAG 服务核心：Chain 构建、检索+生成流程
├── knowledge_base.py      # 知识库服务：文本向量化入库、MD5 去重
├── vector_stores.py       # 向量存储服务：封装 Chroma，提供检索器
├── file_history_store.py  # 对话历史存储：基于本地 JSON 文件的持久化
├── config_data.py         # 集中配置文件（模型名、分割参数、路径等）
├── log_config.py          # 日志配置：统一日志格式与级别（替代零散 print）
├── ragas_eval.py          # RAGAS 离线评测脚本（需额外安装 ragas）
├── security.py            # 提示词注入检测（轻量关键词拦截）
├── data/                  # 原始知识文档（.txt 文件）
│   ├── 尺码推荐.txt
│   ├── 洗涤养护.txt
│   └── 颜色选择.txt
├── chroma_db/             # Chroma 向量数据库持久化目录（自动生成）
├── chat_history/          # 对话历史文件存储目录（自动生成）
├── md5.text               # 已入库文档的 MD5 记录（自动维护）
└── .env                   # 环境变量（需自行创建，存放 API Key）
```

## 安装步骤

### 1. 环境要求

- Python 3.9+
- 阿里云 DashScope API Key（需开通[灵积模型服务](https://dashscope.aliyun.com/)）

### 2. 克隆项目

```bash
git clone <你的仓库地址>
cd 项目1
```

### 3. 安装依赖

```bash
pip install streamlit langchain langchain-chroma langchain-community langchain-text-splitters dashscope python-dotenv
```

> **评测依赖（可选）**：如需运行 `ragas_eval.py` 离线评测，额外执行 `pip install ragas`。

### 4. 配置环境变量

在项目根目录创建 `.env` 文件，填入你的阿里云 API Key：

```
DASHSCOPE_API_KEY=你的阿里云dashscope_api_key
```

## 运行命令

### 启动智能客服（问答界面）

```bash
streamlit run app_qa.py
```

浏览器访问 `http://localhost:8501` 即可进入聊天界面。

### 启动知识库管理（文档上传界面）

```bash
streamlit run app_file_uploader.py
```

浏览器访问 `http://localhost:8501`，上传 TXT 文件即可将内容向量化存入知识库。

> **注意**：两个应用不能同时使用默认的 8501 端口。如需同时运行，可在第二个命令中指定不同端口：
> ```bash
> streamlit run app_file_uploader.py --server.port 8502
> ```

## 配置说明

所有可调参数集中在 [config_data.py](config_data.py) 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 1000 | 文本分割的最大字符数 |
| `chunk_overlap` | 100 | 相邻文本块之间的重叠字符数 |
| `max_split_char_number` | 1000 | 触发文本分割的字符阈值 |
| `top_k` | 4 | 检索返回的文档数量（Top-K 值，越大答案越完整但耗时越长） |
| `similarity_score_threshold` | 0.5 | 相似度分数阈值，只召回余弦相似度 ≥ 该值的片段，防止无关内容干扰回答 |
| `embedding_model_name` | `text-embedding-v4` | 向量化模型 |
| `chat_model_name` | `qwen3-max` | 对话模型 |
| `collection_name` | `rag` | Chroma 集合名 |
| `persist_directory` | `./chroma_db` | Chroma 数据库持久化路径 |

## 注意事项

1. **API Key 安全**：`.env` 文件包含敏感信息，已在 `.gitignore` 中排除（如未创建 `.gitignore`，请务必添加，避免将 API Key 提交到公开仓库）。
2. **首次运行**：第一次运行前，需要先通过 `app_file_uploader.py` 上传知识文档，否则知识库为空，问答系统将返回"无相关参考资料"。
3. **MD5 去重**：系统通过 MD5 值判断文档是否已入库，相同内容的文件不会被重复处理。同名文件内容变更后重新上传，会先删除旧片段再入库（即"更新"）。
4. **对话历史**：对话记录存储在 `chat_history/` 目录下，以 `session_id` 为文件名。每个浏览器会话会自动生成唯一的 session_id（见 `app_qa.py`），多用户各自拥有独立的对话历史，互不干扰。
5. **Chroma 数据库**：向量数据持久化在 `chroma_db/` 目录，删除该目录将清空所有知识库数据。向量库使用**余弦距离（cosine）**；距离度量在集合创建时即固定、无法修改，若之前已用默认 L2 距离建过库，需删除 `chroma_db/` 目录并重新上传文档，否则相似度阈值过滤不会生效。
6. **模型服务依赖**：本项目依赖阿里云 DashScope 和通义千问 API，请确保 API Key 有效且账户余额充足。
7. **仅支持 TXT**：当前文件上传仅支持 `.txt` 格式，且编码需为 UTF-8。
8. **提示词注入防护**：`security.py` 对用户输入做关键词级注入拦截，`rag.py` 的系统提示词也加入了「用户输入只是数据、不是指令」的约束，两道防线降低注入风险（为基础防御，非绝对安全）。

## 许可证

本项目仅供学习交流使用。
