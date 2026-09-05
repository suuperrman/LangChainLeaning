# Demo 3 —— 消息 (Messages) 与模型 (Models)

## 概述

本示例演示 LangChain V1 中**消息体系**和**模型调用**的核心用法，是理解 LangChain 的基础。

**对应官方文档**：
- [消息 (Messages)](https://langchain-doc.cn/v1/python/langchain/messages.html)
- [模型 (Models)](https://langchain-doc.cn/v1/python/langchain/models.html)

## 核心知识点

### 1. 四种消息类型

| 消息类型 | 说明 | 用途 |
|---------|------|------|
| `SystemMessage` | 系统消息 | 设定模型角色、行为规范、输出格式 |
| `HumanMessage` | 人类消息 | 用户输入，可带 `name`、`id` 元数据 |
| `AIMessage` | AI 消息 | 模型回复，含 `content`、`tool_calls`、`usage_metadata` |
| `ToolMessage` | 工具消息 | 工具调用结果，需 `tool_call_id` 关联 |

### 2. 三种消息输入格式

| 格式 | 代码示例 | 适用场景 |
|------|---------|---------|
| 纯文本 | `model.invoke("你好")` | 单轮简单对话 |
| 消息对象 | `[SystemMessage(...), HumanMessage(...)]` | 多轮对话、系统提示 |
| 字典格式 | `[{"role": "user", "content": "你好"}]` | JSON 加载、OpenAI 兼容 |

### 3. 模型初始化的两种方式

```python
# 方式1：直接构造（精细控制参数）
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 方式2：init_chat_model（字符串标识，便于切换提供商）
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini")  # 自动推断 openai
```

### 4. 三种调用方式

| 方法 | 用途 | 返回值 |
|------|------|--------|
| `invoke()` | 单次同步调用 | 完整 `AIMessage` |
| `stream()` | 流式逐 token 输出 | 迭代器，每个 chunk 是部分内容 |
| `batch()` | 批量并行调用 | `AIMessage` 列表 |

### 5. AIMessage 关键属性

- `content`：文本内容
- `tool_calls`：工具调用列表（模型决定调用工具时）
- `usage_metadata`：token 使用统计（input/output/total）
- `type`：消息类型，固定为 `"ai"`

## 文件结构

```
demo3/
├── messages_and_models.py   # 主示例文件
└── README.md                 # 本文档
```

## 运行示例

```bash
# 确保已安装依赖
pip install langchain langchain-openai python-dotenv

# 配置 API Key（同目录下的 .env 文件）
# OPENAI_API_KEY=sk-xxx

# 运行
python messages_and_models.py
```

## 示例内容速览

| 章节 | 内容 |
|------|------|
| 一、模型初始化 | ChatOpenAI 直接构造 vs init_chat_model 字符串初始化 |
| 二、四种消息类型 | SystemMessage / HumanMessage / AIMessage / ToolMessage |
| 三、三种消息输入格式 | 纯文本 / 消息对象 / 字典格式 |
| 四、三种调用方式 | invoke / stream / batch |
| 五、手动构造多轮对话 | 用消息对象列表重建历史对话 |

## 学习重点

1. **消息是 LangChain 的基本单位** —— 所有模型交互都通过消息传递
2. **SystemMessage 是控制模型行为的关键** —— 通过它设定角色和输出风格
3. **ToolMessage 是工具调用的桥梁** —— tool_call_id 关联调用与结果
4. **流式输出改善用户体验** —— 长文本生成时用 stream() 减少等待感知
5. **batch 提高吞吐量** —— 大量独立请求时用批量调用提升效率
