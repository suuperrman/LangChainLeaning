# Demo 4 —— 工具 (Tools)

## 概述

本示例演示 LangChain V1 中**工具（Tools）**的定义与使用。工具是智能体与外部世界交互的桥梁，让 LLM 能够调用 API、查询数据库、执行计算等。

**对应官方文档**：
- [工具 (Tools)](https://langchain-doc.cn/v1/python/langchain/tools.html)

## 核心知识点

### 1. @tool 装饰器

定义工具最简单的方式。函数的**类型提示**定义输入架构，**文档字符串**成为工具描述（帮助模型理解何时使用）。

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """搜索客户数据库，返回匹配的记录。"""
    return f"找到 {limit} 条关于 '{query}' 的记录"
```

### 2. 自定义工具属性

```python
@tool("weather_checker", description="查询指定城市的天气信息。")
def get_weather(city: str) -> str:
    """获取天气信息。"""
    ...
```

- 第一个参数：自定义工具名称
- `description`：覆盖自动生成的描述

### 3. 高级架构定义（Pydantic Schema）

使用 Pydantic 模型定义复杂的输入参数，支持验证、默认值、枚举等。

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    location: str = Field(description="城市名称")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius")
    include_forecast: bool = Field(default=False)

@tool(args_schema=WeatherInput)
def get_detailed_weather(location, units, include_forecast):
    ...
```

### 4. ToolRuntime —— 运行时上下文

工具可以通过 `ToolRuntime` 参数访问运行时信息，**该参数对模型隐藏**，不会出现在工具 schema 中。

| 属性 | 说明 |
|------|------|
| `runtime.state` | 代理状态（messages、自定义字段） |
| `runtime.context` | 不可变上下文（用户ID、配置等） |
| `runtime.store` | 跨对话持久存储（长期记忆） |
| `runtime.stream_writer` | 工具执行时流式传输自定义更新 |
| `runtime.tool_call_id` | 当前工具调用的 ID |

### 5. Command —— 工具更新状态

使用 `langgraph.types.Command` 从工具返回状态更新。

```python
from langgraph.types import Command

@tool
def set_user_preference(key: str, value: str, runtime: ToolRuntime) -> Command:
    """设置用户偏好。"""
    current = runtime.state.get("user_preferences", {})
    current[key] = value
    return Command(update={"user_preferences": current})
```

### 6. 工具与智能体结合

将工具列表传给 `create_agent`，智能体自动在 ReAct 循环中决定何时调用哪个工具。

```python
agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather, calculate],
    system_prompt="你是一个生活助手。",
)
```

## 文件结构

```
demo4/
├── 06_tools.py     # 主示例文件
└── README.md        # 本文档
```

## 运行示例

```bash
# 确保已安装依赖
pip install langchain langchain-openai langgraph pydantic python-dotenv

# 配置 API Key
# OPENAI_API_KEY=sk-xxx

# 运行
python tools.py
```

## 示例内容速览

| 章节 | 内容 |
|------|------|
| 一、基础工具定义 | `@tool` 装饰器、类型提示、docstring 描述 |
| 二、自定义工具属性 | 自定义名称、自定义描述 |
| 三、高级架构定义 | Pydantic Schema、枚举、默认值、字段验证 |
| 四、工具与智能体结合 | ReAct 模式、自动工具选择、多工具调用 |
| 五、ToolRuntime | 访问 state/context/store/stream_writer |
| 六、Command | 工具更新代理状态 |

## 学习重点

1. **类型提示是必需的** —— 它们定义工具的输入架构，模型据此生成正确的调用参数
2. **docstring 很重要** —— 成为工具的描述，直接影响模型是否选择调用该工具
3. **ToolRuntime 对模型隐藏** —— 工具可以访问敏感的运行时信息，而模型看不到这些参数
4. **Command 用于状态更新** —— 工具不仅能返回结果，还能修改代理的状态
5. **工具是智能体的行动能力** —— 没有工具的智能体只是一个聊天机器人
