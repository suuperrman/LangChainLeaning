"""
LangChain V1 入门示例 09 —— 流式传输 (Streaming)

对应文档：
  - https://langchain-doc.cn/v1/python/langchain/streaming.html

核心知识点：
  1. 为什么需要流式传输：改善用户体验，减少等待感知
  2. stream_mode="updates"  : 代理进度（每步一个事件）
  3. stream_mode="messages" : LLM 令牌（逐 token 流式输出）
  4. stream_mode="custom"   : 自定义更新（工具中发出任意数据）
  5. 多模式流式：stream_mode=["updates", "custom"]
  6. get_stream_writer() / runtime.stream_writer : 工具中发送自定义流

三种流式模式对比：
  ┌────────────┬─────────────────────────┬──────────────────────┐
  │ 模式        │ 用途                    │ 输出内容              │
  ├────────────┼─────────────────────────┼──────────────────────┤
  │ updates    │ 展示代理执行进度        │ 每步的完整消息        │
  │ messages   │ 展示 LLM 逐字输出       │ 每个 token 的内容块   │
  │ custom     │ 工具执行时的自定义反馈  │ 任意用户定义数据      │
  └────────────┴─────────────────────────┴──────────────────────┘
"""

import time
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

load_dotenv()

# ============================================================
# 准备工具
# ============================================================
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """查询指定城市的天气信息。"""
    writer = runtime.stream_writer

    # 流式传输自定义进度更新
    writer(f"🔍 正在查询 {city} 的天气数据...")
    time.sleep(0.3)  # 模拟网络延迟
    writer(f"📡 连接气象服务器成功")
    time.sleep(0.3)
    writer(f"📊 正在解析天气数据...")
    time.sleep(0.3)

    weather_data = {
        "北京": "25°C，晴朗，湿度45%",
        "上海": "22°C，多云，湿度60%",
        "广州": "28°C，小雨，湿度80%",
        "深圳": "27°C，晴转多云，湿度65%",
    }
    result = weather_data.get(city, f"{city}：暂无天气数据")

    writer(f"✅ 查询完成")
    return result


@tool
def calculate(expression: str, runtime: ToolRuntime) -> str:
    """执行数学计算。"""
    writer = runtime.stream_writer
    writer(f"🧮 正在计算: {expression}")
    time.sleep(0.2)
    try:
        result = eval(expression)
        writer(f"✅ 计算完成，结果: {result}")
        return f"{expression} = {result}"
    except Exception as e:
        writer(f"❌ 计算错误: {e}")
        return f"错误: {e}"


print("已准备工具: get_weather, calculate")
print("(工具中使用 runtime.stream_writer 发送自定义进度)")
print()

# ============================================================
# 一、stream_mode="updates" —— 代理进度流式传输
# ============================================================
print("=" * 60)
print("一、stream_mode='updates' —— 代理进度")
print("=" * 60)
print("每个代理步骤后发出一个事件（LLM节点/工具节点）")
print("适合：展示代理思考过程、每步结果概览")
print()

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather, calculate],
    system_prompt="你是一个生活助手，可以查询天气和进行计算。",
)

question = "北京的天气怎么样？"
print(f"用户问题: {question}")
print("-" * 40)

step_count = 0
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="updates",
):
    for node_name, data in chunk.items():
        step_count += 1
        last_msg = data["messages"][-1]
        msg_type = type(last_msg).__name__

        if msg_type == "AIMessage":
            if last_msg.tool_calls:
                tc = last_msg.tool_calls[0]
                print(f"[步骤 {step_count}] 🤖 节点: {node_name}")
                print(f"         决定调用工具: {tc['name']}({tc['args']})")
            else:
                print(f"[步骤 {step_count}] 🤖 节点: {node_name}")
                print(f"         最终回答: {last_msg.content[:60]}...")
        elif msg_type == "ToolMessage":
            print(f"[步骤 {step_count}] 🔧 节点: {node_name}")
            print(f"         工具结果: {last_msg.content[:60]}...")
        print()

print(f"共执行 {step_count} 个步骤")
print()

# ============================================================
# 二、stream_mode="messages" —— LLM 令牌流式传输
# ============================================================
print("=" * 60)
print("二、stream_mode='messages' —— LLM 令牌流式输出")
print("=" * 60)
print("LLM 生成每个 token 时都发出，实现逐字显示效果")
print("适合：聊天界面、实时展示生成过程")
print()

question2 = "用三句话介绍一下 LangChain 是什么。"
print(f"用户问题: {question2}")
print("-" * 40)

token_count = 0
current_node = None
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": question2}]},
    stream_mode="messages",
):
    node = metadata.get("langgraph_node", "?")

    # 节点切换时打印分隔
    if node != current_node:
        if current_node is not None:
            print()  # 换行
        current_node = node
        print(f"\n[节点: {node}] ", end="", flush=True)

    # 只打印文本内容（跳过工具调用块等）
    if token.content_blocks:
        for block in token.content_blocks:
            if block.get("type") == "text":
                print(block["text"], end="", flush=True)
                token_count += 1

print(f"\n\n共输出约 {token_count} 个文本 token")
print()

# ============================================================
# 三、stream_mode="custom" —— 自定义流式更新
# ============================================================
print("=" * 60)
print("三、stream_mode='custom' —— 自定义流式更新")
print("=" * 60)
print("工具通过 runtime.stream_writer 发出任意自定义数据")
print("适合：展示工具执行进度、长任务的中间状态")
print()

question3 = "查询深圳的天气。"
print(f"用户问题: {question3}")
print("-" * 40)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": question3}]},
    stream_mode="custom",
):
    print(f"  📡 {chunk}")

print()
print("说明：custom 模式只输出工具中 stream_writer 发送的内容")
print("不包含 LLM 生成的文本，也不包含工具最终结果")
print()

# ============================================================
# 四、多模式流式传输
# ============================================================
print("=" * 60)
print("四、多模式流式传输 (stream_mode 列表)")
print("=" * 60)
print("可以同时接收多种模式的流式输出")
print("格式: for stream_mode, chunk in agent.stream(..., stream_mode=[...])")
print()

question4 = "计算 256 * 1024 等于多少？"
print(f"用户问题: {question4}")
print("-" * 40)

for stream_mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": question4}]},
    stream_mode=["updates", "custom"],
):
    if stream_mode == "custom":
        print(f"  [custom] 📡 {chunk}")
    elif stream_mode == "updates":
        for node_name, data in chunk.items():
            last_msg = data["messages"][-1]
            msg_type = type(last_msg).__name__
            if msg_type == "AIMessage" and last_msg.tool_calls:
                tc = last_msg.tool_calls[0]
                print(f"  [updates] 🤖 {node_name}: 调用工具 {tc['name']}")
            elif msg_type == "ToolMessage":
                print(f"  [updates] 🔧 {node_name}: {last_msg.content[:40]}")
            elif msg_type == "AIMessage":
                print(f"  [updates] 🤖 {node_name}: 最终回答 (省略)")

print()
print("提示：多模式流式让你可以同时展示：")
print("  - 代理的执行进度（updates）")
print("  - 工具的详细过程（custom）")
print("  - LLM 的逐字输出（messages）")
print()

# ============================================================
# 五、禁用流式传输
# ============================================================
print("=" * 60)
print("五、禁用流式传输")
print("=" * 60)
print("某些场景下可能需要禁用流式传输（如多代理系统中控制输出）")
print("可以通过设置 stream=False 或使用特定模型配置禁用")
print()
print("具体方法见模型文档中的 '禁用流式传输' 章节")
