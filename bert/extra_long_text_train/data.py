from datasets import load_dataset

data = load_dataset(path="csv",data_files="/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/extra_long_text_train//data/news/train.csv",split="train")
print(data)
print(type(data))
for i in data:
    print(data["text"])