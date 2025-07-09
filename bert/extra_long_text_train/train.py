#模型训练
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer,AdamW

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义训练的轮次
EPOCH= 30000

token = BertTokenizer.from_pretrained("/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/extra_long_text_train//model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

def collate_fn(data):
    sents = [i[0]for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=1024, # 最大值取决于model中tokenizer_config.json中的model_max_length
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
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    #对加载进来的数据进行编码
    collate_fn=collate_fn
)

val_dataset = MyDataset("validation")
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=2,
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    #对加载进来的数据进行编码
    collate_fn=collate_fn
)

if __name__ == '__main__':
    #开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    #定义优化器
    optimizer = AdamW(model.parameters())
    #定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    #初始化最佳验证准确率
    best_val_acc = 0.0

    for epoch in range(EPOCH):

        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
            #将数据存放到DEVICE上
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),labels.to(DEVICE)
            #前向计算（将数据输入模型，得到输出）
            out = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            #根据输出，计算损失
            loss = loss_func(out,labels)
            #根据损失，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #每隔5个批次输出训练信息
            if i%5==0:
                out = out.argmax(dim=1)
                acc = (out==labels).sum().item()/len(labels)
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")
        #验证模型(判断是否过拟合)
        #设置为评估模式
        model.eval()
        #不需要模型参与训练
        with torch.no_grad():
            val_acc = 0.0
            val_loss = 0.0
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
                # 将数据存放到DEVICE上
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                    DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
                # 前向计算（将数据输入模型，得到输出）
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # 根据输出，计算损失
                val_loss += loss_func(out, labels)
                out = out.argmax(dim=1)
                val_acc+=(out==labels).sum().item()
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f"验证集：loss:{val_loss},acc：{val_acc}")

            #根据验证准确率保存最优参数
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),"params/best_bert.pth")
                print(f"Epoch:{epoch}:保存最优参数：acc:{best_val_acc}")

        #保存最后一轮参数
        torch.save(model.state_dict(),f"params/last_bert.pth")
        print(epoch,f"Epcot：{epoch}最后一轮参数保存成功！")