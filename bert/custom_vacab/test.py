import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token = BertTokenizer.from_pretrained("/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/custom_vacab/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

def collate_fn(data):
    sents = [i[0]for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=500,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids,attention_mask,token_type_ids,labels

#创建数据集
test_dataset = MyDataset("test")
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=100,
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    #对加载进来的数据进行编码
    collate_fn=collate_fn
)

if __name__ == '__main__':
    acc = 0.0
    total = 0

    #开始测试
    print(DEVICE)
    model = Model().to(DEVICE)
    #加载训练参数
    model.load_state_dict(torch.load("params/3_bert.pth"))
    #开启测试模型
    model.eval()
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        # 将数据存放到DEVICE上
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
            DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
        # 前向计算（将数据输入模型，得到输出）
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        acc += (out==labels).sum().item()
        print(i,(out==labels).sum().item())
        total+=len(labels)
    print(f"test_acc：{acc/total}")