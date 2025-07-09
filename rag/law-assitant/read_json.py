# 读取json文件，utf8编码转中文展示，
# /Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/rag/law-assitant/storage/docstore.json
import json

def load_and_print_json(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # print(data)
    # # 遍历数据并打印中文内容
    # for key, value in data['docstore/data'].items():
    #     # 获取中文标题
    #     full_title = value['__data__']['metadata']['full_title']
    #     # 获取文本内容
    #     text = value['__data__']['text']
    #     print(f"标题：{full_title}")
    #     print(f"内容：{text}\n")
    # 将JSON数据格式化并打印
    formatted_json = json.dumps(data, ensure_ascii=False, indent=4)
    print(formatted_json)

# 使用函数
file_path = '/Users/a1233/yiyan/code/Embracing-AI/AI-learn/AI-learn/rag/law-assitant/storage/docstore.json'  
load_and_print_json(file_path)
