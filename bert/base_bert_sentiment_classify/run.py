# 4.模型测试(这里只有测试没有评估，可以人工做评估)
# ===================================================================================
# 基于BERT的中文情感分析模型测试脚本
# 功能：加载训练好的BERT模型，对用户输入的中文文本进行情感分析（正向/负向）
# ===================================================================================

import torch  # PyTorch深度学习框架，用来加载和运行模型
from net import Model  # 导入我们自定义的BERT模型结构（这个文件应该在同一个目录下）
from transformers import BertTokenizer  # 导入BERT的分词器，用来把中文文本转换成模型能理解的数字

# ===================================================================================
# 第一步：设置运行环境和加载必要组件
# ===================================================================================

# 检查电脑是否有GPU（显卡），如果有就用GPU跑得更快，没有就用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)  # 打印出当前使用的设备（GPU或CPU）

# 加载BERT的中文分词器
# 分词器的作用：把"我很开心"这样的句子变成[101, 2769, 1296, 2458, 2552, 102]这样的数字
# 因为神经网络只能理解数字，不能直接理解文字
token = BertTokenizer.from_pretrained("/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/base_bert_sentiment_classify/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

# 定义情感分析的结果标签
# 模型输出0表示负向评价（差评），输出1表示正向评价（好评）
names = ["负向评价","正向评价"]

# 创建模型实例并放到指定设备上（GPU或CPU）
model = Model().to(DEVICE)

# ===================================================================================
# 第二步：定义数据预处理函数
# ===================================================================================

def collate_fn(data):
    """
    数据预处理函数：把用户输入的文本转换成BERT模型能理解的格式
    
    参数：
        data: 用户输入的原始文本，比如"这个产品很好用"
    
    返回：
        input_ids: 文本对应的数字编码
        attention_mask: 告诉模型哪些位置是真实文本，哪些位置是填充的
        token_type_ids: 用来区分不同句子的标记（在单句分类中都是0）
    """
    sents = []  # 创建一个空列表
    sents.append(data)  # 把用户输入的文本放到列表里
    
    # 使用BERT分词器对文本进行编码
    # 这一步是整个预处理的核心，把文字变成数字
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 要编码的文本列表
        truncation=True,  # 如果文本太长就截断，防止超出模型最大长度
        max_length=500,  # 最大长度设为500个字符
        padding="max_length",  # 如果文本不够长就用特殊标记填充到500
        return_tensors="pt",  # 返回PyTorch张量格式
        return_length=True  # 返回文本长度信息
    )
    
    # 提取编码后的三个重要组件
    input_ids = data["input_ids"]  # 文本的数字编码，最重要的输入
    attention_mask = data["attention_mask"]  # 注意力掩码，告诉模型关注哪些位置
    token_type_ids = data["token_type_ids"]  # 句子类型编码，单句任务中都是0

    return input_ids, attention_mask, token_type_ids

# ===================================================================================
# 第三步：定义测试函数
# ===================================================================================

def test():
    """
    测试函数：加载训练好的模型，循环接收用户输入并进行情感分析
    """
    
    # 加载训练好的模型参数
    # "params/1_bert.pth"是之前训练完成后保存的模型权重文件
    # map_location=DEVICE确保模型参数加载到正确的设备上
    model.load_state_dict(torch.load("params/1_bert.pth",map_location=DEVICE))
    
    # 设置模型为评估模式
    # 这很重要！训练模式和评估模式下模型行为不同
    # 评估模式会关闭dropout等训练时的随机性操作
    model.eval()

    # 开始交互式测试循环
    while True:
        # 等待用户输入要分析的文本
        data = input("请输入测试数据（输入'q'退出）：")
        
        # 如果用户输入'q'就退出程序
        if data == 'q':
            print("测试结束")
            break
        
        # 对用户输入的文本进行预处理，转换成模型需要的格式
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        
        # 把预处理后的数据移动到指定设备（GPU或CPU）
        # 确保数据和模型在同一个设备上，否则会报错
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(DEVICE), \
            token_type_ids.to(DEVICE)

        # 进行模型推理（预测）
        with torch.no_grad():  # 关闭梯度计算，节省内存，加快推理速度
            # 将数据输入模型，得到预测结果
            # out是一个包含两个数字的向量，分别表示负向和正向的概率分数
            out = model(input_ids, attention_mask, token_type_ids)
            
            # 找到概率最大的类别
            # argmax(dim=1)返回最大值的索引：0表示负向，1表示正向
            out = out.argmax(dim=1)
            
            # 根据预测结果输出对应的标签
            print("模型判定：",names[out],"\n")

# ===================================================================================
# 第四步：程序入口
# ===================================================================================

if __name__ == '__main__':
    """
    程序的入口点
    只有直接运行这个脚本时才会执行test()函数
    如果是被其他程序导入则不会自动执行
    """
    test()  # 开始测试

# ===================================================================================
# 使用说明：
# 1. 确保已经训练好模型并保存在"params/1_bert.pth"
# 2. 确保net.py文件中定义了Model类
# 3. 运行脚本后输入中文文本，模型会判断是正向还是负向评价
# 4. 输入'q'退出程序
# 
# 示例：
# 输入："这个商品质量很好，我很满意" -> 输出：正向评价
# 输入："产品有问题，不推荐购买" -> 输出：负向评价
# ===================================================================================