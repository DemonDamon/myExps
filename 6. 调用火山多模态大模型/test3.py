import os
# Install SDK:  pip install 'volcengine-python-sdk[ark]' .
from volcenginesdkarkruntime import Ark 

client = Ark(
    # 模型服务的 Base URL .
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    # Get API Key：https://console.volcengine.com/ark/region:ark+cn-beijing/apikey
    api_key='3fbc40fc-551c-4e91-b9a5-a74efc60eb6a', 
)

completion = client.chat.completions.create(
    # 按需替换 Model ID .
    model = "Doubao-seed-1-6-vision-250815",
    messages = [
        {
            "role": "user",  
            "content": [   
                # 图片信息，希望模型理解的图片
                {"type": "image_url", "image_url": {"url":  "https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png"},},
                # 文本消息，希望模型根据图片信息回答的问题
                {"type": "text", "text": "支持输入图片的模型系列是哪个？"}, 
            ],
        }
    ],
)

print(completion.choices[0].message.content)