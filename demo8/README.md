# Demo 8 —— 短期记忆 (Short-term Memory)

## 概述

本示例演示 LangChain V1 中**短期记忆**的机制和用法。短期记忆让智能体能够记住当前对话中的先前交互，是实现多轮对话的基础。

**对应官方文档**：
- [短期记忆 (Short-term Memory)](https://langchain-doc.cn/v1/python/langchain/short-term-memory.html)

## 核心知识点

### 1. 核心概念

| 概念 | 说明 |
|------|------|
| **短期记忆** | 单个对话/线程内的上下文记忆 |
| **checkpointer** | 状态持久化后端（内存/数据库） |
| **thread_id** | 对话线程标识，区分不同会话 |
| **AgentState** | 代理状态，包含 messages 等字段 |

> **短期记忆 ≠ 长期记忆**
> - 短期记忆：单次对话内的上下文（messages）
> - 长期记忆：跨对话的用户偏好、历史记录（store）

### 2. 最基础的用法

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    checkpointer=InMemorySaver(),  # 启用记忆
)

config: RunnableConfig = {"configurable": {"thread_id": "user_001"}}

# 第 1 轮
agent.invoke({"messages": "我叫张三"}, config)

# 第 2 轮（智能体记得第 1 轮的信息）
agent.invoke({"messages": "我叫什么名字？"}, config)
```

### 3. thread_id 隔离

不同 `thread_id` 的对话完全隔离，互不干扰。这是支持多用户的基础。

```python
config_alice = {"configurable": {"thread_id": "alice"}}
config_bob = {"configurable": {"thread_id": "bob"}}
# Alice 和 Bob 的对话完全独立
```

### 4. 自定义状态 (state_schema)

默认 `AgentState` 只有 `messages` 字段，可通过继承扩展：

```python
class CustomAgentState(AgentState):
    user_id: str
    user_name: str
    preferences: dict

agent = create_agent(..., state_schema=CustomAgentState)
```

### 5. 三种消息管理模式

长对话可能超出 LLM 的上下文窗口，常见解决方案：

| 模式 | 说明 | 特点 |
|------|------|------|
| **删除消息** | 永久移除旧消息 | 最简单，但会丢失信息 |
| **修剪消息** | 调用 LLM 前临时截断 | 不影响存储历史，但仍会丢失信息 |
| **总结消息** | 用 LLM 摘要旧消息 | 保留关键信息，有额外 token 成本 |

#### 删除消息（Delete messages）

使用 `RemoveMessage` 从状态中永久删除消息：

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# 删除指定消息
RemoveMessage(id=msg.id)

# 删除所有消息
RemoveMessage(id=REMOVE_ALL_MESSAGES)
```

#### 修剪消息（Trim messages）

使用 `@before_model` 中间件，在模型调用前临时截断：

```python
@before_model
def trim_messages(state, runtime):
    messages = state["messages"]
    if len(messages) > 10:
        # 保留系统消息 + 最近几条
        trimmed = [messages[0]] + messages[-6:]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]}
    return None
```

#### 总结消息（SummarizationMiddleware）

内置的摘要中间件，在 token 接近上限时自动摘要：

```python
from langchain.agents.middleware import SummarizationMiddleware

SummarizationMiddleware(
    model="gpt-4o-mini",
    max_tokens_before_summary=4000,  # 触发阈值
    messages_to_keep=20,              # 摘要后保留的消息数
)
```

### 6. 工具中访问和修改记忆

工具通过 `ToolRuntime` 访问状态：

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

# 读取状态
@tool
def get_user_info(runtime: ToolRuntime) -> str:
    name = runtime.state.get("user_name", "未知")
    return f"用户姓名: {name}"

# 修改状态（返回 Command）
@tool
def set_user_name(name: str, runtime: ToolRuntime) -> Command:
    return Command(update={"user_name": name})
```

### 7. 生产环境的 checkpointer

| 后端 | 包 | 适用场景 |
|------|-----|---------|
| `InMemorySaver` | `langgraph` | 开发、测试 |
| `PostgresSaver` | `langgraph-checkpoint-postgres` | 生产环境 |
| `RedisSaver` | `langgraph-checkpoint-redis` | 高并发场景 |
| `SqliteSaver` | `langgraph-checkpoint-sqlite` | 单机部署 |

## 文件结构

```
demo8/
├── 10_short_term_memory.py   # 主示例文件
└── README.md                  # 本文档
```

## 运行示例

```bash
# 确保已安装依赖
pip install langchain langchain-openai langgraph python-dotenv

# 配置 API Key
# OPENAI_API_KEY=sk-xxx

# 运行
python short_term_memory.py
```

## 示例内容速览

| 章节 | 内容 |
|------|------|
| 一、基础短期记忆 | checkpointer + thread_id 基本用法 |
| 二、线程隔离 | 不同 thread_id 互不干扰 |
| 三、自定义状态 | state_schema 扩展 AgentState |
| 四、删除消息 | RemoveMessage、@after_model 中间件 |
| 五、修剪消息 | @before_model 中间件临时截断 |
| 六、总结消息 | SummarizationMiddleware 自动摘要 |
| 七、工具读写记忆 | ToolRuntime 读写状态、Command 更新 |

## 学习重点

1. **checkpointer 是记忆的核心** —— 没有它就没有状态持久化
2. **thread_id 是多用户的基础** —— 每个用户一个线程，完全隔离
3. **删除是永久的，修剪是临时的** —— 注意区分两者的影响范围
4. **总结消息是最佳实践** —— 既保留上下文，又不超过窗口限制
5. **工具可以读写状态** —— 通过 ToolRuntime 和 Command 实现
6. **生产环境用数据库后端** —— InMemorySaver 只适合开发测试
