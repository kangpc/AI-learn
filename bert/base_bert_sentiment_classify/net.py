# 2.构建、设计模型

from transformers import BertModel
import torch

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

#加载预训练模型
pretrained = BertModel.from_pretrained("/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/base_bert_sentiment_classify/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(DEVICE)
print(pretrained)
#定义下游任务（增量模型）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #设计全连接网络，实现二分类任务
        self.fc = torch.nn.Linear(768,2) # 768是BERT模型的隐藏层大小，2是分类任务的输出类别数，目前为2分类

    def forward(self,input_ids,attention_mask,token_type_ids):
        #冻结Bert模型的参数，让其不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #增量模型参与训练
        out = self.fc(out.last_hidden_state[:,0])
        return out





