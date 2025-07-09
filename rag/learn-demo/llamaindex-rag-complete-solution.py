"""
llamaindex 完整rag解决方案
todo：
    文档分块
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
import datetime
import os
import hashlib

class CompleteRAGSolution:
    def __init__(self, 
                 docs_path="/mnt/workspace/llamaindex-demo/data/劳动法.pdf",
                 storage_path="/mnt/workspace/llamaindex-demo/rag/storage"):
        """
        完整的RAG解决方案
        
        Args:
            docs_path: 文档路径
            storage_path: 向量存储路径
        """
        self.docs_path = docs_path
        self.storage_path = storage_path
        self.conversation_history = []
        
        # 初始化系统
        self.setup_models()
        self.initialize_index()
        self.setup_chat_engine()
        
    def setup_models(self):
        """初始化模型配置"""
        print("🔧 正在初始化模型...")
        
        # 初始化embedding模型
        self.embed_model = HuggingFaceEmbedding(
            model_name="/mnt/workspace/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        
        # 初始化LLM
        self.llm = HuggingFaceLLM(
            model_name="/mnt/workspace/models/Qwen/qwen1_5_1_8b_chat_qlora_xtuner_merged",
            tokenizer_name="/mnt/workspace/models/Qwen/qwen1_5_1_8b_chat_qlora_xtuner_merged",
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"trust_remote_code": True}
        )
        
        # 设置全局配置
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        print("✅ 模型初始化完成！")
        
    def get_docs_hash(self):
        """计算文档内容的哈希值，用于检测文档是否有变化"""
        try:
            if os.path.isfile(self.docs_path):
                with open(self.docs_path, 'rb') as f:
                    content = f.read()
                return hashlib.md5(content).hexdigest()
            elif os.path.isdir(self.docs_path):
                # 如果是目录，计算所有文件的哈希
                hash_obj = hashlib.md5()
                for root, dirs, files in os.walk(self.docs_path):
                    for file in sorted(files):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'rb') as f:
                            hash_obj.update(f.read())
                return hash_obj.hexdigest()
        except Exception as e:
            print(f"⚠️ 计算文档哈希时出错: {e}")
            return None
            
    def save_metadata(self, docs_hash):
        """保存元数据信息"""
        metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "docs_hash": docs_hash,
            "docs_path": self.docs_path,
            "embed_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "llm_model": "qwen1_5_1_8b_chat_qlora_xtuner_merged"
        }
        
        metadata_path = os.path.join(self.storage_path, "metadata.json")
        try:
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存元数据失败: {e}")
            
    def load_metadata(self):
        """加载元数据信息"""
        metadata_path = os.path.join(self.storage_path, "metadata.json")
        try:
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
            
    def should_rebuild_index(self):
        """判断是否需要重建索引"""
        # 检查存储目录是否存在
        if not os.path.exists(self.storage_path):
            return True, "存储目录不存在"
            
        # 检查关键存储文件是否存在
        required_files = ["default__vector_store.json", "docstore.json", "index_store.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(self.storage_path, file)):
                return True, f"缺少关键文件: {file}"
        
        # 检查文档是否有变化
        current_hash = self.get_docs_hash()
        metadata = self.load_metadata()
        
        if not metadata:
            return True, "无元数据信息"
            
        if current_hash and metadata.get("docs_hash") != current_hash:
            return True, "文档内容已变化"
            
        return False, "索引无需重建"
        
    def build_index(self):
        """构建向量索引"""
        print(f"📁 正在加载文档: {self.docs_path}")
        
        # 加载文档
        if os.path.isfile(self.docs_path):
            documents = SimpleDirectoryReader(input_files=[self.docs_path]).load_data()
        elif os.path.isdir(self.docs_path):
            documents = SimpleDirectoryReader(input_dir=self.docs_path).load_data()
        else:
            raise FileNotFoundError(f"文档路径不存在: {self.docs_path}")
            
        print(f"📄 已加载 {len(documents)} 个文档")
        print("🔧 正在构建向量索引（计算文档嵌入）...")
        
        # 构建索引
        self.index = VectorStoreIndex.from_documents(documents)
        
        # 持久化存储
        print(f"💾 正在保存索引到: {self.storage_path}")
        self.index.storage_context.persist(persist_dir=self.storage_path)
        
        # 保存元数据
        docs_hash = self.get_docs_hash()
        self.save_metadata(docs_hash)
        
        print("✅ 向量索引构建并保存完成！")
        
    def load_index(self):
        """从存储加载向量索引"""
        print(f"📚 正在从本地存储加载向量索引: {self.storage_path}")
        storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
        self.index = load_index_from_storage(storage_context)
        print("✅ 向量索引加载完成！")
        
    def initialize_index(self):
        """智能初始化索引（构建或加载）"""
        print("🔍 正在检查向量索引状态...")
        
        should_rebuild, reason = self.should_rebuild_index()
        
        if should_rebuild:
            print(f"🚧 需要重建索引: {reason}")
            self.build_index()
        else:
            print(f"✅ {reason}，直接加载现有索引")
            self.load_index()
            
        # 显示索引信息
        metadata = self.load_metadata()
        if metadata:
            print(f"📊 索引信息: 创建时间 {metadata.get('created_at', '未知')}")
            
    def setup_chat_engine(self):
        """设置聊天引擎"""
        self.chat_engine = self.index.as_chat_engine(
            similarity_top_k=3  # 检索前3个最相似的文档
        )
        
    def save_conversation(self, user_input, bot_response):
        """保存对话历史"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history.append({
            "timestamp": timestamp,
            "user": user_input,
            "bot": str(bot_response)
        })

    def show_conversation_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("📝 暂无对话历史")
            return
        
        print("\n" + "="*60)
        print("📚 对话历史记录")
        print("="*60)
        for i, conv in enumerate(self.conversation_history, 1):
            print(f"\n轮次 {i} [{conv['timestamp']}]")
            print(f"👤 用户: {conv['user']}")
            print(f"🤖 助手: {conv['bot']}")
            print("-" * 40)

    def show_help(self):
        """显示帮助信息"""
        print("\n" + "="*60)
        print("🔧 可用命令")
        print("="*60)
        print("📝 /history   - 查看对话历史")
        print("🔄 /reset     - 重置对话会话")
        print("🔧 /rebuild   - 强制重建向量索引")
        print("📊 /info      - 显示系统信息")
        print("❓ /help      - 显示帮助信息")
        print("👋 /quit      - 退出系统")
        print("💡 直接输入问题即可开始对话")
        print("="*60)

    def show_system_info(self):
        """显示系统信息"""
        print("\n" + "="*60)
        print("📊 系统信息")
        print("="*60)
        print(f"📁 文档路径: {self.docs_path}")
        print(f"💾 存储路径: {self.storage_path}")
        print(f"💬 对话轮次: {len(self.conversation_history)}")
        
        metadata = self.load_metadata()
        if metadata:
            print(f"🕐 索引创建时间: {metadata.get('created_at', '未知')}")
            print(f"🔧 嵌入模型: {metadata.get('embed_model', '未知')}")
            print(f"🤖 语言模型: {metadata.get('llm_model', '未知')}")
        
        print("="*60)

    def reset_conversation(self):
        """重置对话会话"""
        self.chat_engine.reset()
        self.conversation_history.clear()
        print("🔄 对话会话已重置！")
        
    def rebuild_index(self):
        """强制重建索引"""
        print("🔄 正在强制重建向量索引...")
        self.build_index()
        # 重新设置聊天引擎
        self.setup_chat_engine()
        # 重置对话历史
        self.reset_conversation()
        print("✅ 索引重建完成，聊天引擎已更新！")

    def chat(self, user_input):
        """处理用户输入并生成回答"""
        print("🤔 正在思考中...")
        
        # 使用聊天引擎进行对话
        response = self.chat_engine.chat(user_input)
        
        # 保存对话历史
        self.save_conversation(user_input, response)
        
        return response

    def run(self):
        """运行主程序"""
        print("\n" + "="*80)
        print("🎯 完整 RAG 解决方案")
        print("="*80)
        print("💡 智能文档问答助手 - 支持自动索引构建与多轮对话")
        print("🔧 自动检测向量存储状态，智能选择构建或加载模式")
        print("📖 支持多轮对话，可以参考上下文进行回答")
        print("❓ 输入 /help 查看所有可用命令")
        print("="*80)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 请输入您的问题（或输入命令）: ").strip()
                
                # 检查是否为空输入
                if not user_input:
                    print("⚠️ 请输入有效的问题或命令")
                    continue
                
                # 处理系统命令
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command in ['/quit', '/q']:
                        print("👋 感谢使用完整RAG解决方案！再见！")
                        break
                    elif command in ['/help', '/h']:
                        self.show_help()
                        continue
                    elif command == '/history':
                        self.show_conversation_history()
                        continue
                    elif command == '/reset':
                        self.reset_conversation()
                        continue
                    elif command == '/rebuild':
                        self.rebuild_index()
                        continue
                    elif command == '/info':
                        self.show_system_info()
                        continue
                    else:
                        print(f"❌ 未知命令: {user_input}")
                        print("💡 输入 /help 查看可用命令")
                        continue
                
                # 处理正常对话
                response = self.chat(user_input)
                
                # 显示回答
                print(f"\n🤖 助手: {response}")
                
                # 显示简要统计
                print(f"💬 对话轮次: {len(self.conversation_history)}")
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到 Ctrl+C，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("🔄 请重试或输入 /help 查看帮助")

def main():
    """主函数 - 可配置文档和存储路径"""
    
    # 可以根据需要修改这些路径
    docs_path = "/mnt/workspace/llamaindex-demo/data/劳动法.pdf"  # 文档路径
    storage_path = "/mnt/workspace/llamaindex-demo/rag/storage"   # 存储路径
    
    # 创建并运行完整RAG解决方案
    rag_solution = CompleteRAGSolution(docs_path=docs_path, storage_path=storage_path)
    rag_solution.run()

if __name__ == "__main__":
    main() 