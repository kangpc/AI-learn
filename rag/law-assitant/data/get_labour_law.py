# 获取劳动法所有条款
from playwright.sync_api import sync_playwright
import re
import json

def get_labor_law(url):
    with sync_playwright() as p:
        # 启动无头浏览器
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        # 导航到目标页面
        page.goto(url)
        
        # 等待页面加载完成，特别是等待特定元素加载完成
        page.wait_for_selector('div.TRS_PreAppend')
        
        # 获取劳动法的内容
        content_element = page.query_selector('div.TRS_PreAppend')
        if content_element:
            # 获取内容并清理HTML标签
            content = content_element.inner_html()
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = clean_content.replace('\xa0', '').replace('\u3000', '')
            
            # 按段落分割内容
            paragraphs = clean_content.split('\n')
            
            # 过滤掉空段落
            filtered_paragraphs = [para.strip() for para in paragraphs if para.strip()]
            
            # 合并段落，确保条款编号正确
            result = []
            current_chapter = ""
            for para in filtered_paragraphs:
                # 检测章节标题
                if re.match(r'^第\d+章', para):
                    current_chapter = para
                    result.append(f"\n{para}\n")
                else:
                    result.append(para)
            
            # 组合结果
            final_content = "\n".join(result)
            return final_content
        else:
            print("未找到劳动法内容")
            return None


def extract_law_articles(data_str):
    # 正则表达式，匹配每个条款号及其内容
    pattern = re.compile(r'第([一二三四五六七八九十零百]+)条.*?(?=\n第|$)', re.DOTALL)
    # 初始化字典来存储条款号和内容
    lawarticles = {}
    # 搜索所有匹配项
    for match in pattern.finditer(data_str):
        articlenumber = match.group(1)
        articlecontent = match.group(0).replace('第' + articlenumber + '条', '').strip()
        lawarticles[f"中华人民共和国劳动法 第{articlenumber}条"] = articlecontent
    # 转换字典为JSON字符串
    jsonstr = json.dumps(lawarticles, ensure_ascii=False, indent=4)
    return jsonstr

url = "https://www.mohrss.gov.cn/xxgk2020/fdzdgknr/zcfg/fl/202011/t20201102_394625.html"
text = get_labor_law(url)
print(extract_law_articles(text))