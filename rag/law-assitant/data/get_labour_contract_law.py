# 获取劳动合同法所有条款
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
import json

def get_labour_contract_law(url: str) -> str:
    """
    使用 Playwright 无头浏览器加载页面，执行页面 JS 验证，
    然后提取 <div class="TRS_Editor"> 中的所有段落和列表条目文本。
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)  # 可改成 firefox 或 webkit
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            locale="zh-CN"
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle")  # 等待网络空闲，JS 执行完毕
        # 确保 JS 验证脚本执行完成并且页面刷新后，TRS_Editor 已经出现
        page.wait_for_selector("div.TRS_Editor", timeout=10000)
        html = page.content()
        browser.close()

    # 用 BeautifulSoup 提取正文
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", class_="TRS_Editor")
    if not content_div:
        raise RuntimeError("未能在页面中找到 TRS_Editor 容器")

    parts = []
    for elem in content_div.find_all(["p", "li"]):
        text = elem.get_text(strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)


def extract_law_articles(data_str):
    # 正则表达式，匹配每个条款号及其内容
    pattern = re.compile(r'第([一二三四五六七八九十零百]+)条.*?(?=\n第|$)', re.DOTALL)
    # 初始化字典来存储条款号和内容
    lawarticles = {}
    # 搜索所有匹配项
    for match in pattern.finditer(data_str):
        articlenumber = match.group(1)
        articlecontent = match.group(0).replace('第' + articlenumber + '条', '').strip()
        lawarticles[f"中华人民共和国劳动合同法 第{articlenumber}条"] = articlecontent
    # 转换字典为JSON字符串
    jsonstr = json.dumps(lawarticles, ensure_ascii=False, indent=4)
    return jsonstr

if __name__ == "__main__":
    url = "https://www.mohrss.gov.cn/xxgk2020/fdzdgknr/zcfg/fl/202011/t20201102_394622.html"
    
    text = get_labour_contract_law(url)
    print(extract_law_articles(text))
