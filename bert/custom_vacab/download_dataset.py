from datasets import load_dataset

# 加载数据集
dataset = load_dataset("ndiy/ChnSentiCorp")

# 保存到本地以便后续使用
dataset.save_to_disk("/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/bert/custom_vacab/data/ChnSentiCorp")

