import os
import io
import base64
import time
import json
import uuid
import openpyxl
from PIL import Image
from volcenginesdkarkruntime import Ark
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('config.env')

# 修改后的提示词
prompt = """
你是一名专业的题目解析专家，专注于识别图片中的题目信息，并提供准确、详细且全面的答案和解析。你不仅能够精准识别题目内容，而且能够通过用户的提问进行严谨的思考过程，并给出正确的解答和解析。\n主要任务和解决的问题包括：\n理解用户的意图：准确识别用户需要你解答的图片中的题目{query}\n题目识别与解析：能够根据图片中的题目内容，题目中的表格只能使用<table>标签格式输出并且与后面的文字内容区分开，识别出题目并按照题目顺序进行处理。题目中公式格式识别输出限制为LateX格式。\n答案给出：对题目提供正确的答案文本，力求准确无误，答案中公式格式限制为LateX格式。\n解题过程说明：对题目的解题过程进行详细说明，清晰呈现思考逻辑和步骤，便于理解和验证。\n输出要求：最终输出结果必须结构清晰，回答中不包含任何多余的解释性文字、提示词或附加标记，只输出符合要求的内容。表格采用<table>标签形式输出、数学公式采用LateX格式输出。禁止重复输出，当输出的内容重复并且循环重复字数达到300字的时候直接中止输出。\n具体要求：\n 1. 针对用户提问，按照图片中题目的顺序逐题解答。当用户未指定题目范围时候，则将照片中所有的题目叙述一遍，并依次回答所有题目，确保回答题目编号正确，编号禁止出现任何中文字符（例如 T1、T2、…）。\n 2. 题目的回答必须包含两个部分：答案和解析过程，每个解析过程需要提供两个及以上的解题思路。\n 3. 答案部分需要简明扼要地给出题目的正确答案；解析过程部分则需要详细描述解题的思路、步骤和相关逻辑。\n 4. 输出时禁止添加任何解释性文字，只需严格按照上述格式输出题目的答案及详细解析。\n 5. 最终输出内容必须符合以上所有要求，且结构严谨、答案准确全面。答案输出表格只能采用<table> 输出并且和后面的文字内容区分开、数学公式采用LateX格式输出。\n 6. 题目中有表格和图形信息时，保留表格图形的完整信息。\n 7. 请开始执行任务，并确保每道题目的输出都严格遵循上述格式。禁止重复解答，当输出的答案或者题目内容重复并且循环重复字数达到300字的时候直接中止输出。\n 8. 如果一个大题中有多个小题，请按照格式结构化输出，例如T1，其中包含的小题用(1)，(2)、(3)等表示，格式一致，同时T1大题中的完整题目也要保留并解析出来。\n 9. 在输出换行符时务必仅使用\n来表示。\n请严格按照下列标准格式输出题目的答案和解析过程，格式输出标准如下，严格按照以下标准格式输出，禁止输出json格式：\n### T1:\n\n### 识别题目：\n\n这是题目的文本\n\n### 答案:\n\n这是题目的答案\n\n### 解析过程:\n\n这是题目解析过程说明\n\n### T2:\n\n### 识别题目：\n\n这是题目的文本\n\n### 答案:\n\n这是题目的答案\n\n### 解析过程:\n\n这是题目解析过程说明
"""

def encode_image(img, max_size=2000):
    width, height = img.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    byte_stream = io.BytesIO()
    img_format = img.format or 'JPEG'
    save_params = {'format': img_format}
    if img_format.upper() == 'JPEG':
        save_params['quality'] = 100
    img.save(byte_stream, **save_params)
    return base64.b64encode(byte_stream.getvalue()).decode('utf-8'), img_format.lower()

# 从环境变量读取配置参数
CONCURRENT_NUM = int(os.getenv('CONCURRENT_NUM', '10'))  # 并发数
STREAM_TIMEOUT = int(os.getenv('STREAM_TIMEOUT', '300'))  # 流式输出超时时间（秒）
API_KEY = os.getenv('ARK_API_KEY')  # 从环境变量读取
BASE_URL = os.getenv('ARK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
MODEL_NAME = os.getenv('MODEL_NAME', 'doubao-seed-1-6-vision-250815')

# 目标图片
TARGET_IMAGE = '/Users/damon/myWork/myExps/6. 调用火山多模态大模型/typical/1.jpg'

# 输出Excel文件路径
OUTPUT_EXCEL = '/Users/damon/myWork/myExps/6. 调用火山多模态大模型/并发测试结果.xlsx'

# 线程安全的打印锁
print_lock = Lock()

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs)

def process_single_request(task_id, image_path, base64_image, img_format, prompt_text):
    """
    处理单个请求
    :param task_id: 任务UUID
    :param image_path: 图片路径
    :param base64_image: base64编码的图片
    :param img_format: 图片格式
    :param prompt_text: 提示词
    :return: dict包含结果信息
    """
    result = {
        '输入id': task_id,
        '图片路径': image_path,
        '输入提示词': prompt_text[:100] + '...' if len(prompt_text) > 100 else prompt_text,  # 截取前100字符
        '输出结果': '',
        '状态': '失败',
        '总耗时(秒)': 0,
        '首Token时间(秒)': 0,
        '生成耗时(秒)': 0,  # 生成内容的时间（总耗时-首Token时间）
        '输出字符数': 0,
        '生成速度(字符/秒)': 0,
        '错误信息': ''
    }
    
    try:
        # 初始化客户端（每个线程独立）
        client = Ark(
            base_url=BASE_URL,
            api_key=API_KEY
        )
        
        safe_print(f"[{task_id[:8]}] 🚀 开始处理...")
        
        s_time = time.time()
        
        # 使用 Ark SDK 发送请求
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/{img_format};base64,{base64_image}"
                        }},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            stream=True
        )
        
        all_content = ""
        chunk_count = 0
        first_token_time = 0
        
        # 设置超时
        start_time = time.time()
        
        for chunk in response:
            # 检查超时
            if time.time() - start_time > STREAM_TIMEOUT:
                result['错误信息'] = f'超时（>{STREAM_TIMEOUT}秒）'
                safe_print(f"[{task_id[:8]}] ⏰ 超时！")
                return result
            
            if chunk.choices:
                if chunk_count == 0:
                    first_token_time = time.time() - s_time
                    result['首Token时间(秒)'] = round(first_token_time, 2)
                    safe_print(f"[{task_id[:8]}] ⚡ 首Token: {first_token_time:.2f}秒")
                
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content = delta.content
                    all_content += content
                chunk_count += 1
        
        total_time = time.time() - s_time
        generation_time = total_time - first_token_time  # 纯生成时间
        char_count = len(all_content)
        gen_speed = char_count / generation_time if generation_time > 0 else 0
        
        result['总耗时(秒)'] = round(total_time, 2)
        result['生成耗时(秒)'] = round(generation_time, 2)
        result['输出字符数'] = char_count
        result['生成速度(字符/秒)'] = round(gen_speed, 2)
        result['输出结果'] = all_content
        result['状态'] = '成功'
        
        safe_print(f"[{task_id[:8]}] ✅ 完成！总耗时: {total_time:.2f}秒 | 生成: {generation_time:.2f}秒 | {char_count}字符 | 速度: {gen_speed:.2f}字符/秒")
        
    except Exception as e:
        result['错误信息'] = str(e)
        safe_print(f"[{task_id[:8]}] ❌ 错误: {e}")
    
    return result

def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f"🔥 并发测试开始")
    print(f"{'='*80}")
    print(f"📊 配置信息：")
    print(f"  - 并发数: {CONCURRENT_NUM}")
    print(f"  - 超时时间: {STREAM_TIMEOUT}秒")
    print(f"  - 模型: {MODEL_NAME}")
    print(f"  - 目标图片: {TARGET_IMAGE}")
    print(f"{'='*80}\n")
    
    # 读取图片并编码（只编码一次）
    img = Image.open(TARGET_IMAGE)
    file_size = os.path.getsize(TARGET_IMAGE) / 1024  # KB
    
    print(f"📷 图像信息：")
    print(f"  - 文件大小: {file_size:.2f} KB")
    print(f"  - 图像尺寸: {img.size[0]} x {img.size[1]} 像素")
    print(f"  - 图像格式: {img.format}")
    
    base64_image, img_format = encode_image(img, 2000)
    encoded_size = len(base64_image) / 1024  # KB
    print(f"  - Base64编码后: {encoded_size:.2f} KB\n")
    
    # 创建任务列表
    tasks = []
    for i in range(CONCURRENT_NUM):
        task_id = str(uuid.uuid4())
        tasks.append((task_id, TARGET_IMAGE, base64_image, img_format, prompt))
    
    print(f"🚀 启动 {CONCURRENT_NUM} 个并发任务...\n")
    
    # 使用线程池执行
    results = []
    overall_start = time.time()
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_NUM) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_request, *task): task[0] 
            for task in tasks
        }
        
        # 收集结果
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                safe_print(f"[{task_id[:8]}] 💥 异常: {e}")
                results.append({
                    '输入id': task_id,
                    '图片路径': TARGET_IMAGE,
                    '输入提示词': prompt[:100] + '...',
                    '输出结果': '',
                    '状态': '异常',
                    '总耗时(秒)': 0,
                    '首Token时间(秒)': 0,
                    '生成耗时(秒)': 0,
                    '输出字符数': 0,
                    '生成速度(字符/秒)': 0,
                    '错误信息': str(e)
                })
    
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"✅ 所有任务完成！总耗时: {overall_time:.2f}秒")
    print(f"{'='*80}\n")
    
    # 统计信息
    success_count = sum(1 for r in results if r['状态'] == '成功')
    fail_count = len(results) - success_count
    
    print(f"📊 统计信息：")
    print(f"  - 成功: {success_count}/{CONCURRENT_NUM}")
    print(f"  - 失败: {fail_count}/{CONCURRENT_NUM}")
    
    if success_count > 0:
        success_results = [r for r in results if r['状态'] == '成功']
        avg_total_time = sum(r['总耗时(秒)'] for r in success_results) / success_count
        avg_first_token = sum(r['首Token时间(秒)'] for r in success_results) / success_count
        avg_gen_time = sum(r['生成耗时(秒)'] for r in success_results) / success_count
        avg_chars = sum(r['输出字符数'] for r in success_results) / success_count
        avg_speed = sum(r['生成速度(字符/秒)'] for r in success_results) / success_count
        
        print(f"\n  ⏱️  平均时间：")
        print(f"     - 首Token: {avg_first_token:.2f}秒")
        print(f"     - 纯生成: {avg_gen_time:.2f}秒")
        print(f"     - 总耗时: {avg_total_time:.2f}秒")
        print(f"\n  📝 平均输出：")
        print(f"     - 字符数: {avg_chars:.0f}")
        print(f"     - 生成速度: {avg_speed:.2f} 字符/秒")
        
        # 显示性能差异
        max_time = max(r['总耗时(秒)'] for r in success_results)
        min_time = min(r['总耗时(秒)'] for r in success_results)
        print(f"\n  📊 性能范围：")
        print(f"     - 最快: {min_time:.2f}秒")
        print(f"     - 最慢: {max_time:.2f}秒")
        print(f"     - 差异: {max_time - min_time:.2f}秒 ({((max_time/min_time - 1) * 100):.1f}%)")
    
    # 保存到Excel
    print(f"\n💾 正在保存结果到: {OUTPUT_EXCEL}")
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "并发测试结果"
    
    # 写入表头
    headers = ['输入id', '图片路径', '输入提示词', '输出结果', '状态', '总耗时(秒)', '首Token时间(秒)', 
               '生成耗时(秒)', '输出字符数', '生成速度(字符/秒)', '错误信息']
    ws.append(headers)
    
    # 写入数据
    for result in results:
        ws.append([
            result['输入id'],
            result['图片路径'],
            result['输入提示词'],
            result['输出结果'],
            result['状态'],
            result['总耗时(秒)'],
            result['首Token时间(秒)'],
            result['生成耗时(秒)'],
            result['输出字符数'],
            result['生成速度(字符/秒)'],
            result['错误信息']
        ])
    
    # 调整列宽
    ws.column_dimensions['A'].width = 38  # UUID
    ws.column_dimensions['B'].width = 50  # 图片路径
    ws.column_dimensions['C'].width = 30  # 提示词
    ws.column_dimensions['D'].width = 100  # 输出结果
    ws.column_dimensions['E'].width = 10  # 状态
    ws.column_dimensions['F'].width = 15  # 总耗时
    ws.column_dimensions['G'].width = 18  # 首Token时间
    ws.column_dimensions['H'].width = 18  # 生成耗时
    ws.column_dimensions['I'].width = 15  # 输出字符数
    ws.column_dimensions['J'].width = 20  # 生成速度
    ws.column_dimensions['K'].width = 50  # 错误信息
    
    wb.save(OUTPUT_EXCEL)
    print(f"✅ 结果已保存！\n")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()