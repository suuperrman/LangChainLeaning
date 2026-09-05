# Demo 7 —— 流式传输 (Streaming)

## 概述

本示例演示 LangChain V1 中**流式传输**的三种模式和高级用法。流式传输能显著改善用户体验，让用户实时看到输出过程而不是等待完整结果。

**对应官方文档**：
- [流式传输 (Streaming)](https://langchain-doc.cn/v1/python/langchain/streaming.html)

## 核心知识点

### 1. 三种流式模式对比

| 模式 | `stream_mode` 值 | 用途 | 输出粒度 |
|------|-----------------|------|---------|
| **代理进度** | `"updates"` | 展示代理执行的每一步 | 每个节点一条完整消息 |
| **LLM 令牌** | `"messages"` | 展示 LLM 逐字生成过程 | 每个 token 的内容块 |
| **自定义更新** | `"custom"` | 工具执行时的自定义进度 | 任意用户定义数据 |

### 2. stream_mode="updates" —— 代理进度

每个代理步骤后发出一个事件（LLM 节点 / 工具节点）。

```python
for chunk in agent.stream(input, stream_mode="updates"):
    for node_name, data in chunk.items():
        last_msg = data["messages"][-1]
        # last_msg 可能是 AIMessage（工具调用或最终回答）或 ToolMessage
```

**典型输出序列**：
1. `model` 节点 → `AIMessage`（含 tool_calls）
2. `tools` 节点 → `ToolMessage`（工具执行结果）
3. `model` 节点 → `AIMessage`（最终回答）

### 3. stream_mode="messages" —— LLM 令牌

LLM 生成每个 token 时都发出，实现打字机效果。

```python
for token, metadata in agent.stream(input, stream_mode="messages"):
    node = metadata["langgraph_node"]  # 当前节点名
    # token.content_blocks 包含文本块、工具调用块等
    for block in token.content_blocks:
        if block.get("type") == "text":
            print(block["text"], end="", flush=True)
```

### 4. stream_mode="custom" —— 自定义更新

工具通过 `runtime.stream_writer` 发出任意自定义数据。

```python
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    writer = runtime.stream_writer
    writer(f"正在查询 {city} 的天气...")
    # ... 执行查询
    writer("查询完成")
    return result
```

```python
for chunk in agent.stream(input, stream_mode="custom"):
    print(chunk)  # 打印工具发出的自定义消息
```

### 5. 多模式流式传输

可以同时接收多种模式的流式输出：

```python
for stream_mode, chunk in agent.stream(
    input,
    stream_mode=["updates", "custom"]
):
    if stream_mode == "updates":
        ...
    elif stream_mode == "custom":
        ...
```

### 6. 禁用流式传输

某些场景（如多代理系统）可能需要禁用流式传输，可通过模型配置实现。

## 文件结构

```
demo7/
├── 09_streaming.py   # 主示例文件
└── README.md          # 本文档
```

## 运行示例

```bash
# 确保已安装依赖
pip install langchain langchain-openai langgraph python-dotenv

# 配置 API Key
# OPENAI_API_KEY=sk-xxx

# 运行
python 09_streaming.py
```

## 示例内容速览

| 章节 | 内容 |
|------|------|
| 准备工具 | get_weather、calculate（带 stream_writer） |
| 一、代理进度流式 | `stream_mode="updates"`，逐节点展示 |
| 二、LLM 令牌流式 | `stream_mode="messages"`，逐 token 输出 |
| 三、自定义流式更新 | `stream_mode="custom"`，工具内发送进度 |
| 四、多模式流式 | `stream_mode=["updates", "custom"]` |
| 五、禁用流式传输 | 概念介绍 |

## 学习重点

1. **三种模式各有用途** —— updates 看进度、messages 看逐字、custom 看工具过程
2. **updates 是按节点的** —— 每个完整步骤发一次，适合展示思考过程
3. **messages 是按 token 的** —— 最细粒度，聊天界面必备
4. **custom 需要工具配合** —— 通过 `runtime.stream_writer` 发出数据
5. **多模式可组合** —— 同时接收多种流，构建丰富的交互体验
6. **流式传输改善 UX** —— 即使总时间相同，渐进式显示感觉更快
