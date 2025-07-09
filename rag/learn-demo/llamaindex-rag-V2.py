"""
使用llamaindex实现rag，并使用本地存储持久化
todo：
    文档分块
    多轮对话
"""
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM

# 初始化embedding模型
embed_model = HuggingFaceEmbedding(
    model_name="/mnt/workspace/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# 设置全局的embed_model属性
Settings.embed_model = embed_model

# 使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(
    model_name="/mnt/workspace/models/Qwen/qwen1_5_1_8b_chat_qlora_xtuner_merged",
    tokenizer_name="/mnt/workspace/models/Qwen/qwen1_5_1_8b_chat_qlora_xtuner_merged",
    model_kwargs={"trust_remote_code": True},
    tokenizer_kwargs={"trust_remote_code": True}
)

# 设置全局的llm属性
Settings.llm = llm

print("正在从本地存储加载向量索引...")

# 从本地存储加载已经持久化的向量索引
storage_context = StorageContext.from_defaults(persist_dir="/mnt/workspace/llamaindex-demo/rag/storage")
index = load_index_from_storage(storage_context)

print("向量索引加载完成！")

# 创建查询引擎
query_engine = index.as_query_engine()

# 测试查询 - 相同的问题
print("\n=== 测试查询 1：xtuner是什么？ ===")
response1 = query_engine.query("xtuner是什么？")
print(f"回答：{response1}")

# 测试查询 - 不同的问题
print("\n=== 测试查询 2：它支持哪些模型？ ===")
response2 = query_engine.query("XTuner支持哪些模型？")
print(f"回答：{response2}")
