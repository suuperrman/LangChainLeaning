# Demo 5 —— 智能体 (Agents) 基础

## 概述

本示例演示 LangChain V1 中**智能体（Agents）**的核心概念和基础用法。智能体 = LLM + 工具 + 循环推理，是 LangChain 最核心的抽象。

**对应官方文档**：
- [智能体 (Agents)](https://langchain-doc.cn/v1/python/langchain/agents.html)

## 核心知识点

### 1. create_agent —— 创建智能体

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",          # 模型（推理引擎）
    tools=[tool1, tool2],         # 工具列表（行动能力）
    system_prompt="你是一个助手。",  # 系统提示（行为规范）
)
```

`create_agent` 基于 LangGraph 构建了一个**基于图**的智能体运行时，是生产就绪的实现。

### 2. 核心组件

| 组件 | 作用 | 类比 |
|------|------|------|
| 模型 (Model) | 推理引擎，决定做什么 | 大脑 |
| 工具 (Tools) | 行动能力，执行具体操作 | 手 |
| 系统提示 (System Prompt) | 行为规范，角色设定 | 性格 |
| ReAct 循环 | 推理-行动-观察的迭代过程 | 思考过程 |

### 3. ReAct 循环

智能体遵循 **ReAct（Reasoning + Acting）** 模式：

```
用户问题
   ↓
┌─────────────────────────────┐
│  1. 推理 (Thought)           │
│     模型思考需要做什么        │
│  ↓                           │
│  2. 行动 (Action)            │
│     选择工具并调用            │
│  ↓                           │
│  3. 观察 (Observation)       │
│     获取工具执行结果          │
└──────────────┬──────────────┘
               ↓
         还需要继续吗？
          ↙       ↘
       是          否
        ↓           ↓
    继续循环     输出最终答案
```

### 4. 静态模型 vs 动态模型

- **静态模型**：创建时固定，最常用
  ```python
  agent = create_agent(model="gpt-4o-mini", tools=tools)
  ```

- **动态模型**：运行时根据状态选择（用 `@wrap_model_call` 中间件）
  - 长对话用高级模型，短对话用基础模型（成本优化）
  - 根据任务类型路由到不同模型

### 5. 智能体状态 (AgentState)

智能体的状态是一个字典，默认包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | List[BaseMessage] | 对话历史消息列表 |

可通过 `state_schema` 参数扩展自定义字段：

```python
class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(..., state_schema=CustomAgentState)
```

### 6. 系统提示

- **字符串方式**：简单直接，创建时固定
- **动态方式**：用 `@dynamic_prompt` 中间件根据上下文生成（见中间件示例）

## 文件结构

```
demo5/
├── 07_agents_intro.py   # 主示例文件
└── README.md             # 本文档
```

## 运行示例

```bash
# 确保已安装依赖
pip install langchain langchain-openai langgraph python-dotenv

# 配置 API Key
# OPENAI_API_KEY=sk-xxx

# 运行
python 07_agents_intro.py
```

## 示例内容速览

| 章节 | 内容 |
|------|------|
| 一、准备工具 | 定义 3 个模拟工具（搜索产品、查库存、算价格） |
| 二、创建基础智能体 | `create_agent` 基本用法，静态模型 + 工具 |
| 三、ReAct 循环 | 多步工具调用的完整过程演示 |
| 四、智能体状态 | `AgentState` 结构、messages 字段 |
| 五、动态系统提示 | 概念介绍（详见中间件示例） |
| 六、动态模型选择 | `@wrap_model_call` 概念介绍 |
| 七、空工具列表 | 无工具的智能体 = 带状态管理的 LLM |

## 学习重点

1. **智能体 = LLM + 工具 + 循环** —— 核心是 ReAct 模式
2. **模型决定做什么，工具负责执行** —— 职责分离
3. **状态是智能体的记忆** —— `AgentState` 保存所有上下文
4. **多步推理是关键能力** —— 复杂问题需要多次工具调用和推理
5. **空工具也有用** —— 统一的状态管理、中间件、流式传输支持
