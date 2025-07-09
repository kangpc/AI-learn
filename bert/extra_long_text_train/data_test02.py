import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

#读取CSV文件
csv_file_path = "/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/extra_long_text_train/data/Weibo/train.csv"
df = pd.read_csv(csv_file_path)

#定义重采样策略
#如果想要过采样，使用RandomOverSampler
#如果想要欠采样，使用RandomUnderSampler
#我们在这里使用RandomUnderSampler进行欠采样
#random_state控制随机数生成器的种子数，一般给42
rus = RandomUnderSampler(sampling_strategy="auto",random_state=42)

#将特征和标签分开
X = df[["text"]]
Y = df[["label"]]
print(X)
print(Y)

#应用重采样
X_resampled,Y_resampled = rus.fit_resample(X,Y)
print(f"X_resampled: {X_resampled}")
print(f"Y_resampled: {Y_resampled}")
#合并特征和标签，创建系的DataFrame
df_resampled = pd.concat([X_resampled,Y_resampled],axis=1)

print(f"df_resampled: {df_resampled}")

#保存均衡数据到新的csv文件
df_resampled.to_csv("/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/extra_long_text_train/data/Weibo/new_train.csv",index=False)