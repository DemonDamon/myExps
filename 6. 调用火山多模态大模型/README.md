# 火山多模态大模型调用项目

## 📦 依赖安装

```bash
pip install python-dotenv openpyxl pillow volcengine-python-sdk[ark] requests
```

## 🔑 配置说明

### 1. 复制配置文件

项目使用 `config.env` 文件管理敏感信息。该文件已包含所有配置，直接使用即可。

### 2. 环境变量说明

`config.env` 文件包含以下配置：

```env
# 火山引擎 API 配置
ARK_API_KEY=主API密钥
ARK_API_KEY_BACKUP=备用API密钥
ARK_BASE_URL=API地址

# 模型配置
MODEL_NAME=主模型名称
MODEL_NAME_VISION_PRO=高级视觉模型名称

# GMP-SAAS 接口配置（备用）
GMP_API_URL=GMP接口地址
GMP_API_AK=GMP访问密钥
GMP_API_SK=GMP密钥

# 并发测试配置
CONCURRENT_NUM=并发数量
STREAM_TIMEOUT=流式输出超时时间（秒）
```

### 3. 修改配置

如需修改配置，直接编辑 `config.env` 文件中的对应值即可。

## 📂 文件说明

- **test.py** - 主测试文件，支持多线程并发测试，输出Excel结果
- **test2.py** - 简单的API调用示例（使用requests）
- **test3.py** - SDK调用示例（使用Ark SDK）
- **config.env** - 环境变量配置文件（包含API密钥等敏感信息）
- **.gitignore** - Git忽略文件（确保敏感信息不被提交）

## 🚀 使用方法

### test.py - 并发测试

```bash
python test.py
```

功能特点：
- ✅ 多线程并发调用
- ✅ 流式输出，实时显示结果
- ✅ 详细性能统计（首Token时间、生成速度等）
- ✅ 结果自动保存到Excel

配置参数（在 `config.env` 中修改）：
- `CONCURRENT_NUM` - 并发数量（默认10）
- `STREAM_TIMEOUT` - 超时时间（默认300秒）

### test2.py - 简单API调用

```bash
python test2.py
```

使用场景：测试基础API调用功能

### test3.py - SDK调用示例

```bash
python test3.py
```

使用场景：展示如何使用Ark SDK进行简单的图像识别

## 📊 输出结果

test.py 会生成 `并发测试结果.xlsx`，包含以下字段：
- 输入id (UUID)
- 图片路径
- 输入提示词
- 输出结果
- 状态
- 总耗时(秒)
- 首Token时间(秒)
- 生成耗时(秒)
- 输出字符数
- 生成速度(字符/秒)
- 错误信息

## ⚠️ 注意事项

1. **不要提交 `config.env` 文件到Git仓库**（已在.gitignore中配置）
2. **API密钥保密**：不要在公开场合分享API密钥
3. **并发限制**：根据API配额合理设置并发数量
4. **超时设置**：大图片或复杂任务可能需要更长的超时时间

## 🔧 故障排查

### 问题1：提示找不到 config.env
**解决方案**：确保 `config.env` 文件在项目根目录下

### 问题2：API调用失败
**解决方案**：
1. 检查 `config.env` 中的API密钥是否正确
2. 确认API密钥有访问权限
3. 检查网络连接

### 问题3：导入错误
**解决方案**：安装缺失的依赖
```bash
pip install python-dotenv
```

## 📝 开发说明

所有敏感信息都应该存储在 `config.env` 文件中，不要在代码中硬编码API密钥。

使用方式：
```python
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('config.env')

# 读取配置
API_KEY = os.getenv('ARK_API_KEY')
```

