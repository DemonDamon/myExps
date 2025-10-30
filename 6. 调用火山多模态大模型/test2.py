import requests
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('config.env')

# 从环境变量读取配置
API_KEY = os.getenv('ARK_API_KEY_BACKUP')
BASE_URL = os.getenv('ARK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
MODEL_NAME = os.getenv('MODEL_NAME_VISION_PRO', 'doubao-1.5-vision-pro-250328')

url = f"{BASE_URL}/chat/completions"

payload = json.dumps({
   "stream": False,
   "model": MODEL_NAME,
   "messages": [
      {
         "role": "system",
         "content": "###思考逻辑\n请分析输入图像并按以下逻辑处理：\n首先检测图像中是否存在任何形式的可识别文字（包括印刷体、手写体等）\n  存在文字 → 设置\"是否有文字\"为\"是\"\n  无文字 → 设置\"是否有文字\"为\"否\"，\"语言类型\"为\"无\"\n当存在文字时，识别文字的主要语言类型：\n若语言属于以下列表：中文、英语、日语、韩语、法语、泰语、葡萄牙语、意大利语、西班牙语、俄语、丹麦语、荷兰语、马来语、瑞典语、印尼语、波兰语、罗马尼亚语、土耳其语、希腊语、匈牙利语、越南语、印地语 → 输出对应语言名称\n非列表中的语言 → 输出\"其他\"\n\n###输出要求\n1.最终输出必须是严格合法的 JSON 对象；\n2.输出必须为纯 JSON，不得包含多余字符、解释性文字、代码块符号（如 ```）或转义符；\n3.不得将整个 JSON 作为字符串输出（例如禁止出现 \\n、\\\"、\"{...}\" 形式）；\n输出示例：\n1.未识别到文字：\n{\n  \"是否有文字\": \"否\",\n  \"语言类型\": \"无\"\n}\n2.识别到英语文字：\n{\n  \"是否有文字\": \"是\",\n  \"语言类型\": \"英语\"\n}"
      },
      {
         "role": "user",
         "content": "把图中的文字使用英文输出"
      },
      {
         "role": "user",
         "content": [
            {
               "type": "image_url",
               "image_url": {
                  "url": "https://jinran-one.tos-cn-beijing.volces.com/img_v3_02nl_82630cc6-8b8b-415a-9a1a-8f65f3ba4ccg.jpg"
               }
            }
         ]
      }
   ]
})
headers = {
   'Authorization': f'Bearer {API_KEY}',
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)