"""
使用llamaindex+chroma实现rag 多轮对话
todo：文档分块
"""

import chromadb
from llama_index.core import Settings,SimpleDirectoryReader,VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.agent.workflow import ReActAgent

# ----启动chroma----
# 定义向量数据库
chroma_client = chromadb.PersistentClient()

# 创建集合
# chroma_client.create_collection(name="quickstart")
# # 获取已经存在的向量数据库
# chroma_collection = chroma_client.get_collection(name="quickstart")
# print(chroma_collection) # Collection(name=quickstart)

# 尝试获取集合，如果不存在则创建
chroma_collection = chroma_client.get_or_create_collection(name="Labor-Law")
print("已获取或创建本地知识库：", chroma_collection.name)

# ----包装成ChromaVectorStore----
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# ----加载embedding & LLM
embed_model = HuggingFaceEmbedding(
    # 指定一个预训练的sentence-transformers模型的路径
    model_name="/mnt/workspace/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

llm = HuggingFaceLLM(model_name="/mnt/workspace/models/Qwen/qwen1_5_1_8b_chat_qlora_xtuner_merged",
               tokenizer_name="/mnt/workspace/models/Qwen/qwen1_5_1_8b_chat_qlora_xtuner_merged",
               model_kwargs={"trust_remote_code":True},
               tokenizer_kwargs={"trust_remote_code":True})

# ----用 Settings 统一注入----
# 把 embedding model、LLM、vector store、prompt template……等「模块」
# 都统一注入到全局Settings对象中，然后在构建索引、查询时把它传进去
Settings.embed_model = embed_model
Settings.llm = llm

# ----构建索引 & 聊天 ----
# 加载目录的所有文档
# documents = SimpleDirectoryReader(input_dir="/mnt/workspace/llamaindex-demo").load_data()
# 加载指定文档
documents = SimpleDirectoryReader(input_files=["/mnt/workspace/llamaindex-demo/data/劳动法.pdf"]).load_data()
# 创建一个向量索引VectorStoreIndex,并使用之前加载的文档来构建向量索引
# 此索引将文档转换为向量，并存储这些向量到 ChromaDB 中以用于快速检索
index = VectorStoreIndex.from_documents(documents, vector_store = vector_store)

# 用新的 Agent 构造器，显式传入 max_iterations
agent = ReActAgent(
    llm=llm, 
    max_iterations=5, 
    verbose=True,               # 如果想看每一步日志
    # tools=[...],              # 如果你想给 Agent 额外的工具
)

# chat引擎进行多轮对话
chat_engine = index.as_chat_engine(agent=agent)

# 第一次对话
user_question1 = "劳动法第一章是什么内容？"
resp1 = chat_engine.chat(user_question1)
print(f"user_question1:{user_question1}\nresp1: {resp1}")

# 多轮对话——chat_engine会自动管理对话历史
user_question2 = "关于工资的内容是在哪一章哪一条？"
resp2 = chat_engine.chat(user_question2)
print(f"user_question2:{user_question2}\nresp2: {resp2}")

