# -*- coding: utf-8 -*-
"""
法律条文智能问答系统（基于RAG架构）

==== 文件目标 ====
本文件实现了一个基于检索增强生成（RAG）技术的法律条文智能助手系统，主要功能包括：

1. **数据处理模块**：
   - 加载和验证JSON格式的法律条文数据
   - 将法律条文转换为可检索的文本节点
   - 确保数据完整性和一致性

2. **向量化存储模块**：
   - 使用HuggingFace BGE模型进行文本嵌入
   - 基于ChromaDB构建持久化向量数据库
   - 实现高效的语义相似度检索

3. **智能问答模块**：
   - 集成Ollama本地大语言模型（Qwen）
   - 实现基于语义检索的法律条文问答
   - 提供详细的法律依据和相关度评分

4. **交互界面**：
   - 命令行式的实时问答交互
   - 显示检索到的相关法律条文
   - 提供回答的可信度和来源追溯

==== 技术架构 ====
- 检索模型：BAAI/bge-small-zh-v1.5（中文语义嵌入）
- 生成模型：Qwen 3.0 0.6B（通过Ollama部署）
- 向量数据库：ChromaDB（支持余弦相似度检索）
- 框架：LlamaIndex（RAG应用开发框架）

==== 适用场景 ====
- 法律从业者的条文检索和解释
- 普通用户的法律知识咨询
- 法律教育和培训辅助工具
- 法律文档智能化处理系统

作者：yiyan_k
创建时间：2025年7月4日
版本：v1.0
"""
import json
import time
from pathlib import Path
from typing import List, Dict
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate

# ================== 提示词模板配置 ==================

# 定义问答系统的提示词模板，采用ChatML格式
# 该模板确保AI助手严格基于法律条文进行会回答，避免幻觉和错误信息
QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个专业的法律助手，请严格根据以下法律条文回答问题：\n"
    "相关法律条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# 将模板字符串转换为LlamaIndex可识别的PromptTemplate对象
response_template = PromptTemplate(QA_TEMPLATE)

# ================== 系统配置类 ==================

class Config:
    """
    系统配置管理类
    """
    # 嵌入模型配置
    EMBED_MODEL_PATH = "/Users/a1233/.cache/modelscope/hub/models/BAAI/bge-small-zh-v1.5"
    # 大语言模型
    LLM_MODEL_NAME = "qwen3:1.7b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    # 数据存储路径
    DATA_DIR = "/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/rag/law-assitant/data"
    VECTOR_DB_DIR = "/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/rag/law-assitant/chroma_db"
    PERSIST_DIR = "/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/rag/law-assitant/storage"
    # 检索参数
    COLLECTION_NAME = "labor_laws"
    TOP_K = 3  # 检索时返回的最相关条文数量

# ================== 模型初始化模块 ==================

def init_models():
    """
    初始化嵌入模型和大语言模型
    
    该函数负责：
    1. 加载预训练的中文语义嵌入模型（BGE-small-zh-v1.5）
    2. 初始化Ollama本地大语言模型服务连接
    3. 将模型设置为LlamaIndex的全局默认模型
    4. 执行模型功能验证测试
    
    Returns:
        tuple: (嵌入模型对象, 大语言模型对象)        
    """
    embed_model = HuggingFaceEmbedding(
        model_name = Config.EMBED_MODEL_PATH,
        # encode_kwargs = {
        #     "normalize_embeddings": True,
        #     "device": "cuda" if hasattr(Settings, "device") else "cpu"
        # }
    )

    llm = Ollama(
        model = Config.LLM_MODEL_NAME,
        base_url = Config.OLLAMA_BASE_URL,
        temperature = 0.3, # 生成温度，较低值使输出更确定性
        request_timeout = 120.0, # 请求超时时间（秒），防止长时间等待
        thinking = False
    )

    # 将嵌入模型和大语言模型注入全局Settings中
    Settings.embed_model = embed_model
    Settings.llm = llm

    # 模型功能验证：生成测试嵌入来确保模型正常工作
    try:
        test_embeding = embed_model.get_text_embedding("测试文本")
        print(f"✅ Embedding模型验证成功，向量维度：{len(test_embeding)}")
    except Exception as e:
        raise RuntimeError(f"❌ Embedding模型验证失败：{str(e)}")
    
    return embed_model, llm

# ================== 数据处理模块 ==================

def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """
    加载并验证JSON格式的法律条文数据
    
    该函数负责：
    1. 扫描指定目录下的所有JSON文件
    2. 逐个加载并验证数据格式的正确性
    3. 统一数据结构，添加源文件元信息
    4. 进行数据完整性检查
    
    Args:
        data_dir (str): 包含JSON法律条文文件的目录路径
        
    Returns:
        List[Dict]: 统一格式的法律条文数据列表，每个元素包含：
            - content: 原始法律条文字典
            - metadata: 包含source字段的元信息

    数据格式要求：
        - JSON文件根元素必须是列表
        - 列表中每个元素必须是字典
        - 字典的所有值必须是字符串类型
    """

    # 扫描目录，获取所有JSON文件
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"❌ 未在目录 {data_dir} 中找到任何JSON文件"
    
    print(f"📁 发现 {len(json_files)} 个JSON文件，开始加载...")
    
    all_data = []  # 存储所有加载的数据 
    
    # 逐个处理JSON文件
    for json_file in json_files:
        print(f"  📄 正在处理：{json_file.name}")
        try:
            # 使用UTF-8编码读取文件，确保中文字符正确解析
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 数据结构验证
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 的根元素应为列表，实际为 {type(data).__name__}")
                
                # 验证列表中每个元素的格式
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for idx, item in enumerate(data):
                        if not isinstance(item, dict):
                            raise ValueError(f"文件 {json_file.name} 第 {idx+1} 个元素应为字典，实际为 {type(item).__name__}")
                    
                # 数据格式化：为每个法律条文添加元信息
                file_data = []
                for item in data:
                    formatted_item = {
                        "content": item,  # 原始法律条文内容
                        "metadata": {"source": json_file.name}  # 来源文件信息
                    }
                    file_data.append(formatted_item)
                
                # 将当前文件的数据添加到总数据列表中
                # extend()方法将file_data列表中的所有元素逐个添加到all_data列表末尾
                # 相比于append()添加整个列表作为单个元素，extend()是将列表内容合并
                all_data.extend(file_data)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"❌ JSON解析失败 {json_file}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"❌ 文件处理失败 {json_file}: {str(e)}")
    
    print(f"🎉 数据加载完成，总计 {len(all_data)} 个法律文件条目")
    return all_data


def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """
    将原始法律数据转换为LlamaIndex可处理的TextNode对象
    
    该函数负责：
    1. 解析法律条文的标题和内容
    2. 提取法律名称和具体条款信息
    3. 生成稳定的节点ID，避免重复
    4. 创建包含丰富元信息的TextNode对象
    
    Args:
        raw_data (List[Dict]): 原始法律数据列表，来自load_and_validate_json_files
        
    Returns:
        List[TextNode]: LlamaIndex TextNode对象列表，用于构建向量索引
        
    节点ID生成规则：
        格式："{源文件名}::{法律条文完整标题}"
        这确保了ID的唯一性和稳定性
        
    元信息字段说明:
        - law_name: 法律名称（从标题中提取的第一部分）
        - article: 具体条款（从标题中提取的第二部分）
        - full_title: 完整的法律条文标题
        - source_file: 来源JSON文件名
        - content_type: 固定值"legal_article"，用于内容分类
    """
 
    nodes = []  # 存储生成的TextNode对象
    
    print("🔄 开始创建文本节点...")
    
    total_entries = 0
    processed_entries = 0
    
    # 遍历所有原始数据条目
    for entry_idx, entry in enumerate(raw_data):
        law_dict = entry["content"]  # 法律条文字典
        source_file = entry["metadata"]["source"]  # 来源文件名
        
        total_entries += len(law_dict)
        print(f"  处理第 {entry_idx + 1} 个法律文档，包含 {len(law_dict)} 条条文")

        # 处理单个法律条文字典中的每个条目
        for full_title, content in law_dict.items():
            
            # 生成稳定且唯一的节点ID
            # 使用双冒号分隔符降低标题中出现冲突的概率
            node_id = f"{source_file}::{full_title}"

            # 解析法律条文标题
            # 预期格式："法律名称 具体条款"，例如："劳动法 第十条"
            parts = full_title.split(" ", 1)  # 只分割一次，避免条款中的空格干扰
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"

            # 创建TextNode对象
            node = TextNode(
                text=content,  # 法律条文的具体内容
                id_=node_id,   # 显式设置稳定ID，避免自动生成的随机ID
                metadata={
                    "law_name": law_name,           # 法律名称
                    "article": article,             # 具体条款
                    "full_title": full_title,       # 完整标题
                    "source_file": source_file,     # 来源文件名
                    "content_type": "legal_article"  # 内容类型
                }
            )
            nodes.append(node)
            processed_entries += 1
    
    print(f"📊 统计信息：")
    print(f"  - 总条文数：{total_entries}")
    print(f"  - 已处理：{processed_entries}")
    print(f"  - 创建节点：{len(nodes)}")
    print(f"🎉 成功创建 {len(nodes)} 个文本节点")
    return nodes
    
# ================== 向量存储管理模块 ==================

def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    """
    初始化向量存储系统，构建和加载向量索引

    该函数负责：
    1. 创建或连接ChromaDB持久化向量数据库
    2. 判断是否需要重新构建索引或加载已有索引
    3. 执行向量化和索引构建过程
    4. 验证存储系统的完整性
    5. 提供双重持久化保障
    
    Args:
        nodes (List[TextNode]): 文本节点列表，如果为None则只加载已有索引
        
    Returns:
        VectorStoreIndex: 可用于查询的向量索引对象
    
    工作流程：
        1. 初始化ChromaDB客户端和集合
        2. 检查数据库中是否已有数据
        3. 如果为空且提供了nodes，则构建新索引
        4. 如果已有数据，则加载现有索引
        5. 执行存储完整性验证

    持久化机制：
        - ChromaDB负责向量数据的持久化
        - LlamaIndex负责文档存储和索引结构的持久化
        - 双重保障确保数据安全性
    """

    print("🔧 初始化向量存储系统...")

    # 创建chromadb持久化客户端PersistentClient，确保向量数据持久化到磁盘
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    # 获取或创建ChromaDB集合,使用余弦相似度作为距离度量，适合语义相似度计算
    chroma_collection = chroma_client.get_or_create_collection(
        name = Config.COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"}
    )
    # 创建存储上下文，连接ChromaDB后端
    storage_context = StorageContext.from_defaults(
        vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    )
    # 检查ChromaDB中是否已有数据，决定是构建新索引还是加载已有索引
    existing_count = chroma_collection.count()
    print(f"📊 ChromaDB中现有数据量：{existing_count}")
    
    if existing_count == 0 and nodes is not None:
        # 情况1：数据库为空且提供了新节点，构建新索引
        print(f"🆕 开始构建新的向量索引（{len(nodes)}个节点）...")
        # 将文档节点添加到存储上下文的文档存储中,这是LlamaIndex管理文档的重要步骤
        storage_context.docstore.add_documents(nodes)
        # 构建向量索引
        # show_progress=True 显示向量化进度条
        index = VectorStoreIndex(
            nodes,
            storage_context = storage_context,
            show_progress = True
        )
        # 双重持久化保障
        # 1. 通过存储上下文持久化（文档存储 + 向量存储）
        storage_context.persist(persist_dir = Config.PERSIST_DIR)
        print(f"storage_context: {storage_context}")
        # 2. 通过索引对象持久化（索引结构）
        index.storage_context.persist(persist_dir = Config.PERSIST_DIR)
        print(f"index: {index}")
        print("✅ 新索引构建完成并已持久化")
        
    else:
        # 情况2：数据库中已有数据，加载现有索引
        print("📂 加载已有的向量索引...")
        
        try:
            # 从持久化目录和向量存储中重建存储上下文
            storage_context = StorageContext.from_defaults(
                persist_dir = Config.PERSIST_DIR,
                vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
            )
            # 从向量存储中重建索引
            index = VectorStoreIndex.from_vector_store(
                storage_context.vector_store,
                storage_context = storage_context,
                embed_model = Settings.embed_model
            )
            print("✅ 已有索引加载完成")
        except Exception as e:
            print(f"⚠️ 索引加载失败：{str(e)}")
            if nodes is not None:
                print("🔄 尝试重新构建索引...")
                # 递归调用，重新构建索引
                return init_vector_store(nodes)
            else:
                raise RuntimeError("❌ 无法加载索引且未提供节点数据")
    
    # 存储系统完整性验证
    print("\n🔍 执行存储完整性验证...")
    
    # 检查文档存储状态
    doc_count = len(storage_context.docstore.docs)
    print(f"📚 DocStore中的文档数量：{doc_count}")
    
    if doc_count > 0:
        # 显示文档存储示例
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"📝 示例文档ID：{sample_key}")
        
        # 验证向量存储状态
        vector_count = chroma_collection.count()
        print(f"🔢 向量存储中的条目数量：{vector_count}")
        
        if vector_count != doc_count:
            print(f"⚠️  警告：文档数量({doc_count})与向量数量({vector_count})不匹配")
    else:
        print("❌ 警告：文档存储为空，请检查节点添加逻辑！")
    
    print("✅ 向量存储系统初始化完成\n")
    return index

# ================== 主程序入口 ==================

def main():
    """
    主程序入口函数
   
    负责整个法律助手系统的启动和运行：
    1. 初始化AI模型（嵌入模型 + 大语言模型）
    2. 数据管理（加载/创建向量索引）
    3. 启动交互式问答系统
    4. 处理用户查询并显示结果
    
    程序流程：
        启动 → 模型初始化 → 数据加载 → 索引构建/加载 → 交互式问答 → 退出

    交互命令：
        - 输入法律相关问题：执行RAG检索和回答
        - 输入 'q' 或 'Q'：退出程序
    """

    print("🚀 启动法律条文智能问答系统")
    print("=" * 50)

    # 第一步：初始化AI模型
    print("🤖 初始化AI模型...")
    try:
        embed_model, llm = init_models()
        print("✅ AI模型初始化成功")
    except Exception as e:
        print(f"❌ AI模型初始化失败：{str(e)}")
        return
    
    # 第二步：加载和处理法律数据和索引构建
    print("🔍 加载和处理法律数据和索引构建...")

    # 智能判断是否需要加载原始数据
    # 如果向量数据库目录不存在，说明是首次运行，需要加载数据
    if not Path(Config.VECTOR_DB_DIR).exists():
        print("🔄 首次运行，开始加载原始数据...")
        try:
            # 加载并验证原始法律数据
            raw_data = load_and_validate_json_files(Config.DATA_DIR)
            # 创建文本节点,nodes里面一个node就是一条法律条文
            nodes = create_nodes(raw_data)
        except Exception as e:
            print(f"❌ 数据加载失败：{str(e)}")
            return
    else:
        print("📁 检测到已有数据，跳过原始数据加载")
        nodes = None  # 已有数据时不重新加载

    # 第三步：初始化向量存储
    print("🔧 初始化向量存储...")
    try:
        start_time = time.time()
        index = init_vector_store(nodes)
        elapsed_time = time.time() - start_time
        print(f"⏱️  索引处理耗时：{elapsed_time:.2f}秒")
        print("✅ 向量存储初始化成功")
    except Exception as e:
        print(f"❌ 向量存储初始化失败：{str(e)}")
        return
    
    # 第四步：创建查询引擎
    print("🔍 创建智能查询引擎...")
    # try:
    query_engine = index.as_query_engine(
        similarity_top_k=Config.TOP_K,
        text_qa_template=response_template,
        verbose=True
    )
    print("✅ 查询引擎创建成功")
    # except Exception as e:
    #     print(f"❌ 查询引擎创建失败：{str(e)}")
    #     return
    
    # 第五步：启动交互式问答系统
    print("\n" + "=" * 50)
    print("🎯 法律条文智能助手已准备就绪！")
    print("💡 使用说明：")
    print("   - 请输入您的法律相关问题")
    print("   - 系统将基于法律条文为您提供准确回答")
    print("   - 输入 'q' 退出程序")
    print("=" * 50)
    
    # 交互式问答循环
    question_count = 0  # 问题计数器

    while True:
        try:
            # 获取用户输入
            question = input(f"\n📝 请输入您的问题（第{question_count + 1}个问题，输入q退出）: ").strip()
            # 检查是否要退出
            if question.lower() == 'q':
                print("👋 感谢使用法律条文智能助手，再见！")
                break
            
            # 验证输入是否为空
            if not question:
                print("⚠️  请输入有效的问题")
                continue
            
            question_count += 1
            print(f"\n🔍 正在分析您的问题：{question}")
            
            # 执行RAG查询
            start_time = time.time()
            try:
                response = query_engine.query(question)
                query_time = time.time() - start_time
                
                # 显示AI助手回答
                print(f"\n🤖 智能助手回答：")
                print("-" * 40)
                print(response.response)
                print("-" * 40)
                print(f"⏱️  查询耗时：{query_time:.2f}秒")
                
                # 显示检索到的法律依据
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"\n📚 法律依据（共{len(response.source_nodes)}条）：")
                    
                    for idx, node in enumerate(response.source_nodes, 1):
                        meta = node.metadata
                        print(f"\n[{idx}] {meta['full_title']}")
                        print(f"    📁 来源文件：{meta['source_file']}")
                        print(f"    ⚖️  法律名称：{meta['law_name']}")
                        print(f"    📄 条款内容：{node.text[:100]}{'...' if len(node.text) > 100 else ''}")
                        
                        # 显示相关度得分（如果可用）
                        if hasattr(node, 'score'):
                            print(f"    🎯 相关度得分：{node.score:.4f}")
                else:
                    print("\n⚠️  未找到相关的法律条文")
                    
            except Exception as e:
                print(f"❌ 查询处理失败：{str(e)}")
                print("💡 请尝试重新表述您的问题")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，程序退出")
            break
        except Exception as e:
            print(f"❌ 程序异常：{str(e)}")
            print("🔄 程序继续运行...")
    
    print("\n🏁 程序已退出")

# ================== 程序入口点 ==================

if __name__ == "__main__":
    main()
    