# 查看数据集类别分布是否均衡
import pandas as pd

# df = pd.read_csv("/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/custom_define_vocab/data/Weibo/new_train.csv")
df = pd.read_csv("/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/extra_long_text_train/data/news/train.csv")
#统计每个类别的数据量
category_counts = df["label"].value_counts()

#统计每个类别的比值
total_data = len(df)
category_ratios = (category_counts / total_data) *100

print(category_counts)
print(category_ratios)