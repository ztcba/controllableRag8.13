# web_search.py
from typing import List, TypedDict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import re # 导入正则表达式库用于清理HTML标签
from graphstate import PlanExecute


def run_web_search(state: PlanExecute) -> PlanExecute:
    """
    使用Selenium模拟浏览器从 https://www.science.org/ 爬取最新一期的完整头条标题，
    并将其更新到状态的 'aggregated_context' 字段中。

    参数:
        state (PlanExecute): LangGraph的当前状态。

    返回:
        PlanExecute: 更新了 'aggregated_context' 字段的图状态。
    """
    print("---开始执行网络搜索 (使用Selenium - V2)---")

    # --- Selenium 设置 ---
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/5.3.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # 【修改1】: 解决SSL错误和减少日志输出
    chrome_options.add_argument('--ignore-certificate-errors') # 忽略证书错误
    chrome_options.add_argument('--allow-running-insecure-content') # 允许运行不安全内容
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging']) # 关闭烦人的日志

    service = Service(ChromeDriverManager().install())
    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        url = "https://www.science.org/"
        print(f"正在访问: {url}")
        # 增加页面加载超时时间
        driver.set_page_load_timeout(30)
        driver.get(url)

        # 等待页面元素加载
        time.sleep(3)

        selector = 'h3[class*="grid-hero-teaser"] a'
        
        title_element = driver.find_element(By.CSS_SELECTOR, selector)
        
        if title_element:
            # 【修改2】: 获取'title'属性而不是内部文本
            full_title_html = title_element.get_attribute('title')
            
            # 清理可能存在的HTML标签 (例如 <i>, <b> 等)
            # 使用简单的字符串替换或正则表达式
            cleaned_title = re.sub(r'<[^>]+>', '', full_title_html)

            print(f"成功找到完整标题: {cleaned_title}")
            state['aggregated_context'] = f"找到的文章标题是：{cleaned_title}"
        else:
            error_message = f"无法在页面上找到指定的标题元素 (CSS Selector: '{selector}')。"
            print(error_message)
            state['aggregated_context'] = error_message

    except Exception as e:
        error_message = f"使用Selenium进行爬取时发生错误: {e}"
        print(error_message)
        state['aggregated_context'] = error_message
    finally:
        if driver:
            print("---关闭浏览器---")
            driver.quit()

    return state

