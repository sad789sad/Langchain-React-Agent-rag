# Langchain-React-Agent-rag
基于 LangChain 框架实现的 ReAct Agent，集成 RAG 检索增强、多工具调用、多轮对话、提示词切换。系统能根据用户意图自动判断任务类型（知识问答 / 报告生成），调用合适的工具和知识库完成推理，并通过Streamlit  流式界面实时展示 Agent 的思考与执行过程。 RAG 检索增强：采用混合检索（向量 + BM25）策略；引入DAT策略，实时调整向量与BM25的融合权重；结合 Qwen3‑Reranker精排。
