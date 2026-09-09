# Demo 6 —— 结构化输出 (Structured Output)

## 概述

本示例演示 LangChain V1 中**结构化输出**的各种用法。结构化输出让智能体以可预测的格式返回数据（Pydantic 对象、字典等），无需解析自然语言响应。

**对应官方文档**：
- [结构化输出 (Structured Output)](https://langchain-doc.cn/v1/python/langchain/structured-output.html)

## 核心知识点

### 1. 两种策略

| 策略 | 说明 | 适用模型 | 可靠性 |
|------|------|---------|--------|
| `ProviderStrategy` | 提供商原生结构化输出 | OpenAI、Grok 等 | 最高 |
| `ToolStrategy` | 基于工具调用实现 | 所有支持工具调用的模型 | 高 |

直接传入 schema 类型时，LangChain 会自动选择最佳策略。

### 2. 四种 Schema 定义方式

| 方式 | 示例 | 返回类型 | 特点 |
|------|------|---------|------|
| **Pydantic** | `class Info(BaseModel)` | Pydantic 对象 | 最推荐，验证完整 |
| **Dataclass** | `@dataclass class Info` | dataclass 实例 | 简洁，轻量 |
| **TypedDict** | `class Info(TypedDict)` | 普通 dict | 与类型提示兼容 |
| **JSON Schema** | `{"type": "object", ...}` | 普通 dict | 最灵活，动态定义 |

### 3. 基本用法

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """一个人的联系信息。"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=ContactInfo,  # 自动选择策略
)

result = agent.invoke({"messages": [{"role": "user", "content": "提取..."}]})
result["structured_response"]  # ContactInfo 对象
```

### 4. 错误处理 (handle_errors)

`ToolStrategy` 支持配置错误处理策略：

| 值 | 行为 |
|----|------|
| `True` | 默认，捕获所有错误并重试 |
| `False` | 不处理，异常直接抛出 |
| `str` | 自定义错误消息 |
| `Exception` 类型 | 仅处理指定异常 |
| `callable` | 自定义错误处理函数 |

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors="请提供 1-5 之间的有效整数评分。",
    ),
)
```

### 5. 自定义工具消息内容

`tool_message_content` 参数可自定义生成结构化输出时对话历史中显示的消息，使对话更自然。

```python
ToolStrategy(
    schema=ActionItem,
    tool_message_content="行动事项已成功捕获！",
)
```

### 6. 结构化输出 + 工具结合

智能体可以先调用工具获取数据，再以结构化格式返回结果。这是实际项目中最常见的用法。

```
用户问题 → 调用工具获取数据 → 解析为结构化格式 → 返回结构化对象
```

## 文件结构

```
demo6/
├── 08_structured_output.py   # 主示例文件
└── README.md                  # 本文档
```

## 运行示例

```bash
# 确保已安装依赖
pip install langchain langchain-openai langgraph pydantic python-dotenv

# 配置 API Key
# OPENAI_API_KEY=sk-xxx

# 运行
python structured_output.py
```

## 示例内容速览

| 章节 | 内容 |
|------|------|
| 一、Pydantic 模式 | 最推荐的方式，字段验证、model_dump() |
| 二、Dataclass 模式 | 简洁的 dataclass 定义 |
| 三、TypedDict 模式 | 返回普通字典，类型提示友好 |
| 四、JSON Schema 模式 | 用字典动态定义 schema |
| 五、复杂嵌套结构 | 嵌套 Pydantic 模型 + 列表 + 枚举 |
| 六、ToolStrategy + 错误处理 | 显式指定策略、handle_errors |
| 七、自定义工具消息内容 | tool_message_content 参数 |
| 八、结构化输出 + 工具 | 先调用工具再返回结构化结果 |

## 学习重点

1. **Pydantic 是首选** —— 类型安全、自动验证、功能最完整
2. **优先用 ProviderStrategy** —— 原生支持更可靠，OpenAI 模型自动启用
3. **错误处理很重要** —— 模型可能输出不符合 schema 的数据，配置重试机制
4. **结果在 `structured_response` 字段** —— 不是 `messages[-1].content`
5. **结构化输出可与工具结合** —— 先获取数据再格式化，是 RAG 和 API 集成的常用模式
6. **适用场景广泛** —— 信息抽取、分类任务、API 接口返回、数据验证
