"""
llamaindex 多轮对话 rag系统
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
import datetime

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

print("🚀 正在启动多轮对话 RAG 系统...")
print("正在从本地存储加载向量索引...")

# 从本地存储加载已经持久化的向量索引
storage_context = StorageContext.from_defaults(persist_dir="/mnt/workspace/llamaindex-demo/rag/storage")
index = load_index_from_storage(storage_context)

print("✅ 向量索引加载完成！")

# 创建聊天引擎（支持多轮对话）
chat_engine = index.as_chat_engine()

# 对话历史存储
conversation_history = []

def save_conversation(user_input, bot_response):
    """保存对话历史"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_history.append({
        "timestamp": timestamp,
        "user": user_input,
        "bot": str(bot_response)
    })

def show_conversation_history():
    """显示对话历史"""
    if not conversation_history:
        print("📝 暂无对话历史")
        return
    
    print("\n" + "="*60)
    print("📚 对话历史记录")
    print("="*60)
    for i, conv in enumerate(conversation_history, 1):
        print(f"\n轮次 {i} [{conv['timestamp']}]")
        print(f"👤 用户: {conv['user']}")
        print(f"🤖 助手: {conv['bot']}")
        print("-" * 40)

def show_help():
    """显示帮助信息"""
    print("\n" + "="*60)
    print("🔧 可用命令")
    print("="*60)
    print("📝 /history  - 查看对话历史")
    print("🔄 /reset    - 重置对话会话")
    print("❓ /help     - 显示帮助信息")
    print("👋 /quit     - 退出系统")
    print("💡 直接输入问题即可开始对话")
    print("="*60)

def reset_conversation():
    """重置对话会话"""
    global chat_engine, conversation_history
    chat_engine.reset()
    conversation_history.clear()
    print("🔄 对话会话已重置！")

def main():
    """主对话循环"""
    print("\n" + "="*60)
    print("🎯 多轮对话 RAG 系统已启动")
    print("="*60)
    print("💡 基于 XTuner 文档的智能问答助手")
    print("📖 支持多轮对话，可以参考上下文进行回答")
    print("❓ 输入 /help 查看可用命令")
    print("="*60)
    
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
                
                if command == '/quit' or command == '/q':
                    print("👋 感谢使用！再见！")
                    break
                elif command == '/help' or command == '/h':
                    show_help()
                    continue
                elif command == '/history':
                    show_conversation_history()
                    continue
                elif command == '/reset':
                    reset_conversation()
                    continue
                else:
                    print(f"❌ 未知命令: {user_input}")
                    print("💡 输入 /help 查看可用命令")
                    continue
            
            # 处理正常对话
            print("🤔 正在思考中...")
            
            # 使用聊天引擎进行对话（自动管理上下文）
            response = chat_engine.chat(user_input)
            
            # 显示回答
            print(f"\n🤖 助手: {response}")
            
            # 保存对话历史
            save_conversation(user_input, response)
            
            # 显示对话轮次统计
            print(f"\n💬 对话轮次: {len(conversation_history)}")
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到 Ctrl+C，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("🔄 请重试或输入 /help 查看帮助")

if __name__ == "__main__":
    main() 