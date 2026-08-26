"""
RAGAS 离线评测脚本

用途：
    用 RAGAS 框架量化评估本项目 RAG 管线的质量，单独运行即可得到检索/回答的评分，
    用于对比「调整参数（top_k / 相似度阈值 / chunk 大小 / Prompt）前后」的效果。

评估的 4 个指标（取值 0~1，越接近 1 越好）：
    - faithfulness（忠实度）：回答是否都有检索到的上下文支撑（是否胡编乱造）
    - answer_relevancy（回答相关性）：回答是否紧扣用户问题
    - context_precision（上下文精确率）：召回的片段里有多少真正有用
    - context_recall（上下文召回率）：正确答案所需的片段是否都被召回了

依赖：
    pip install ragas

运行：
    python ragas_eval.py

说明：
    - SAMPLE_CASES 是示例“金标准”评测集（问题 + 参考答案），请按你的知识库内容替换为真实数据；
    - 评测会用本项目已有的通义千问 LLM 与 embedding 作为 ragas 的“评判模型”，会消耗一定 API 调用量；
    - 已针对 ragas 0.4.3 适配（见下方两处“兼容性修复”）。
"""

import json
import sys
import types
import uuid

# ---- 兼容性修复 1：绕过 ragas 0.4.3 的坏导入 ----
# ragas 0.4.3 在导入时会无条件执行
#   from langchain_community.chat_models.vertexai import ChatVertexAI
# 但新版 langchain-community（0.4.x）已移除该模块，导致 `import ragas` 直接报错。
# 本项目并不使用 VertexAI，这里注册一个占位模块，绕开这个坏导入即可。
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # 占位类：仅用于满足 import，实际不会被调用
        pass

    _vertexai_stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from dotenv import load_dotenv
load_dotenv()

import config_data as config
from rag import RagService
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# ---- 兼容性修复 2：用旧版（legacy）指标，而不是新版 collections ----
# 新版 ragas.metrics.collections 只接受 OpenAI 的 InstructorLLM，会拒绝我们用
# LangchainLLMWrapper 包装的通义千问/DashScope；旧版 ragas.metrics 才支持任意 LangChain LLM。
# 因此这里从 ragas.metrics 导入（会打印 DeprecationWarning，可忽略）。
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall


# 金标准评测集：reference 是“参考答案”，供 context_precision / context_recall 使用
SAMPLE_CASES = [
    {
        "question": "针织毛衣应该如何保养?",
        "reference": "建议手洗或用洗衣袋机洗，平铺晾干，避免悬挂和高温烘干，以防变形。",
    },
    {
        "question": "身高170、体重80公斤应该选多大尺码?",
        "reference": "建议选择 L 或 XL 码，具体需结合胸围、肩宽等数据，以尺码表为准。",
    },
    {
        "question": "皮肤偏黄适合穿什么颜色的衣服?",
        "reference": "建议选择暖色调或中性色，如米白、藏青、酒红等，避免过于鲜艳的荧光色。",
    },
]


def collect_samples(rag: RagService):
    """用真实的检索 + 生成管线，为每个问题产出 ragas 需要的四类数据"""
    retriever = rag.vector_service.get_retriever()
    samples = []
    for i, case in enumerate(SAMPLE_CASES):
        question = case["question"]
        reference = case["reference"]

        # 1) 检索：拿上下文片段（查询与向量库不变，结果与链内部检索一致）
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]

        # 2) 生成：走真实链（含 Prompt 模板与历史管理），每个问题独立 session 避免串扰
        session_config = {"configurable": {"session_id": f"eval_{i}_{uuid.uuid4().hex}"}}
        answer = rag.chain.invoke({"input": question}, session_config)

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=reference,
            )
        )
        print(f"[{i + 1}/{len(SAMPLE_CASES)}] 已采集：{question}（上下文 {len(contexts)} 段）")
    return samples


def main():
    rag = RagService()

    # 用项目已有的 LLM / embedding 作为 ragas 的“评判模型”
    evaluator_llm = LangchainLLMWrapper(ChatTongyi(model=config.chat_model_name))
    evaluator_embeddings = LangchainEmbeddingsWrapper(DashScopeEmbeddings(model=config.embedding_model_name))

    samples = collect_samples(rag)
    dataset = EvaluationDataset(samples=samples)

    # 注意：0.4.3 里 llm / embeddings 是指标的必填构造参数，必须在这里显式传入
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    print(f"开始评测，共 {len(samples)} 条样本 ...")
    # 默认单任务超时 180s，对 qwen3-max 偏短（faithfulness 的 NLI 步骤要一次性生成几十条陈述，
    # context_precision/context_recall 要反复调 LLM），会触发 TimeoutError 导致分数记成 NaN。
    # 这里把超时放宽到 600s。
    run_config = RunConfig(timeout=600)
    result = evaluate(dataset, metrics=metrics, run_config=run_config)

    df = result.to_pandas()
    score_cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]

    print("\n===== 逐条评分 =====")
    print(df[["user_input"] + score_cols].to_string(index=False))

    # 保存明细与平均分，方便对比不同参数下的效果
    df.to_csv("eval_results.csv", index=False, encoding="utf-8-sig")
    summary = {c: round(float(df[c].mean()), 4) for c in score_cols}
    with open("eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== 各指标平均分 =====")
    for name, score in summary.items():
        print(f"  {name}: {score}")
    print("\n明细已保存到 eval_results.csv，平均分已保存到 eval_summary.json")


if __name__ == "__main__":
    main()
