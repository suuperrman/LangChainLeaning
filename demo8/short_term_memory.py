"""
LangChain V1 入门示例 10 —— 短期记忆 (Short-term Memory)

对应文档：
  - https://langchain-doc.cn/v1/python/langchain/short-term-memory.html

核心知识点：
  1. 什么是短期记忆：记住单个对话/线程中的先前交互
  2. checkpointer：状态持久化机制（InMemorySaver / PostgresSaver 等）
  3. thread_id：区分不同对话线程
  4. AgentState：代理状态，包含 messages 等字段
  5. 自定义状态：通过 state_schema 扩展 AgentState
  6. 三种常见模式：修剪消息 / 删除消息 / 总结消息
  7. 访问记忆：工具中通过 ToolRuntime 读写状态

注意：短期记忆 ≠ 长期记忆
  - 短期记忆：单次对话内的上下文（messages）
  - 长期记忆：跨对话的用户偏好、历史记录（store）
"""

from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

load_dotenv()

# ============================================================
# 一、最基础的短期记忆
# ============================================================
print("=" * 60)
print("一、最基础的短期记忆（checkpointer + thread_id）")
print("=" * 60)

print("核心概念：")
print("  - checkpointer: 状态持久化后端（内存/数据库）")
print("  - thread_id   : 对话线程标识，区分不同会话")
print("  - AgentState  : 代理状态（含 messages 消息历史）")
print()

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    checkpointer=InMemorySaver(),  # 内存检查点（生产环境用数据库）
    system_prompt="你是一个友善的助手。",
)

# 使用同一个 thread_id 的多次调用会共享记忆
config: RunnableConfig = {"configurable": {"thread_id": "user_alice"}}

# 第 1 轮对话
print("--- 第 1 轮对话 ---")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好！我叫 Alice，是一名前端工程师。"}]},
    config,
)
print(f"AI: {result['messages'][-1].content[:80]}...")
print()

# 第 2 轮对话（智能体应该记得第 1 轮的信息）
print("--- 第 2 轮对话（测试记忆）---")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？我的职业是什么？"}]},
    config,
)
print(f"AI: {result['messages'][-1].content}")
print()

# 第 3 轮对话
print("--- 第 3 轮对话 ---")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "针对我的职业，推荐一个值得学习的技术方向。"}]},
    config,
)
print(f"AI: {result['messages'][-1].content[:100]}...")
print()

# 查看当前状态中的消息数量
print(f"当前对话历史消息数: {len(result['messages'])}")
print()

# ============================================================
# 二、不同 thread_id 互相隔离
# ============================================================
print("=" * 60)
print("二、不同 thread_id 互相隔离")
print("=" * 60)

config_bob: RunnableConfig = {"configurable": {"thread_id": "user_bob"}}

# Bob 的对话（全新的，不知道 Alice 的信息）
print("--- Bob 的对话（新线程）---")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你知道我叫什么名字吗？"}]},
    config_bob,
)
print(f"AI: {result['messages'][-1].content}")
print()
print("说明：不同 thread_id 的对话完全隔离")
print()

# ============================================================
# 三、自定义状态 (state_schema)
# ============================================================
print("=" * 60)
print("三、自定义状态（扩展 AgentState）")
print("=" * 60)

print("默认 AgentState 只有 messages 字段，可以通过继承扩展")
print()


class CustomAgentState(AgentState):
    """自定义代理状态，添加用户信息字段。"""
    user_id: str          # 用户 ID
    user_name: str        # 用户姓名
    preferences: dict     # 用户偏好设置


agent_custom = create_agent(
    model="gpt-4o-mini",
    tools=[],
    state_schema=CustomAgentState,  # 自定义状态模式
    checkpointer=InMemorySaver(),
    system_prompt="你是一个个性化助手，会根据用户偏好调整回答风格。",
)

# 首次调用时传入自定义状态字段
config_custom: RunnableConfig = {"configurable": {"thread_id": "user_charlie"}}

result = agent_custom.invoke(
    {
        "messages": [{"role": "user", "content": "你好，请用简洁的方式回答我的问题。"}],
        "user_id": "u_001",
        "user_name": "Charlie",
        "preferences": {"style": "concise", "language": "zh-CN", "theme": "dark"},
    },
    config_custom,
)
print(f"状态中的 user_id: {result['user_id']}")
print(f"状态中的 user_name: {result['user_name']}")
print(f"状态中的 preferences: {result['preferences']}")
print(f"AI 回复: {result['messages'][-1].content[:80]}...")
print()

# ============================================================
# 四、删除消息（Delete messages）
# ============================================================
print("=" * 60)
print("四、删除消息（Delete messages）")
print("=" * 60)

print("使用 RemoveMessage 从状态中删除消息")
print("REMOVE_ALL_MESSAGES 可删除全部消息")
print()

from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """删除旧消息，只保留最近的 4 条（2轮对话）。"""
    messages = state["messages"]
    if len(messages) > 4:
        # 删除最早的两条消息（1轮用户 + 1轮AI）
        to_remove = [RemoveMessage(id=m.id) for m in messages[:2]]
        print(f"   删除了 {len(to_remove)} 条旧消息（当前共 {len(messages)} 条）")
        return {"messages": to_remove}
    return None


agent_trim = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
    system_prompt="你是一个简短回答助手。",
)

config_trim: RunnableConfig = {"configurable": {"thread_id": "trim_test"}}

print("进行 5 轮对话，每超过 4 条消息就删除最早的 2 条:")
for i in range(5):
    result = agent_trim.invoke(
        {"messages": [{"role": "user", "content": f"第 {i+1} 条消息：你好"}]},
        config_trim,
    )
    msg_count = len(result["messages"])
    print(f"  第 {i+1} 轮后消息数: {msg_count}")

print()
print("注意：删除消息是永久的，删除的内容无法恢复")
print()

# ============================================================
# 五、修剪消息（Trim messages）
# ============================================================
print("=" * 60)
print("五、修剪消息（Trim messages）")
print("=" * 60)

print("修剪 vs 删除的区别：")
print("  - 删除：永久移除消息")
print("  - 修剪：只在调用 LLM 前临时截断（不影响存储的历史）")
print()

from langchain.agents.middleware import before_model


@before_model
def trim_recent_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """只保留第一条系统消息和最后3条消息，避免上下文过长。"""
    messages = state["messages"]
    if len(messages) <= 4:
        return None

    # 保留第一条 + 最后3条
    trimmed = [messages[0]] + list(messages[-3:])
    # 用 RemoveMessage + 新消息列表替换
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed,
        ]
    }


print("修剪消息使用 @before_model 中间件")
print("在模型调用前临时减少消息数量")
print("（完整示例见 SummarizationMiddleware）")
print()

# ============================================================
# 六、总结消息（SummarizationMiddleware）
# ============================================================
print("=" * 60)
print("六、总结消息（SummarizationMiddleware）")
print("=" * 60)

print("内置的摘要中间件，在 token 接近上限时自动摘要历史消息")
print("比简单删除更好，因为保留了关键信息")
print()

from langchain.agents.middleware import SummarizationMiddleware

agent_summary = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=2000,  # 2000 token 时触发摘要
            messages_to_keep=6,              # 摘要后保留最近 6 条消息
        )
    ],
    checkpointer=InMemorySaver(),
    system_prompt="你是一个聊天助手。",
)

print("SummarizationMiddleware 配置:")
print("  - 触发阈值: 2000 tokens")
print("  - 保留消息数: 最近 6 条")
print("  - 摘要模型: gpt-4o-mini")
print()
print("工作原理：")
print("  1. 每次调用前计算消息历史的 token 数")
print("  2. 超过阈值时，对较早的消息进行摘要")
print("  3. 用摘要 + 最近 N 条消息替换原历史")
print("  4. 既保留上下文，又不超过上下文窗口")
print()

# ============================================================
# 七、在工具中访问和修改记忆
# ============================================================
print("=" * 60)
print("七、在工具中访问和修改记忆")
print("=" * 60)

print("工具通过 ToolRuntime 访问状态（记忆）：")
print("  - 读取: runtime.state['字段名']")
print("  - 写入: 返回 Command(update={...})")
print()

from langgraph.types import Command


class MemoryDemoState(AgentState):
    user_name: str = ""
    favorite_color: str = ""


@tool
def remember_favorite_color(color: str, runtime: ToolRuntime) -> Command:
    """记住用户喜欢的颜色。

    Args:
        color: 颜色名称
    """
    return Command(update={"favorite_color": color})


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """获取当前用户的信息（姓名、偏好颜色等）。"""
    state = runtime.state
    name = state.get("user_name", "未知")
    color = state.get("favorite_color", "未设置")
    return f"用户信息：\n  姓名: {name}\n  喜欢的颜色: {color}"


agent_memory_tool = create_agent(
    model="gpt-4o-mini",
    tools=[remember_favorite_color, get_user_info],
    state_schema=MemoryDemoState,
    checkpointer=InMemorySaver(),
    system_prompt="你是一个记忆助手，可以记住用户的偏好信息。",
)

config_mem: RunnableConfig = {"configurable": {"thread_id": "memory_demo"}}

# 先设置信息
print("--- 设置用户信息 ---")
result = agent_memory_tool.invoke(
    {
        "messages": [{"role": "user", "content": "我叫 Dave，我喜欢蓝色。"}],
        "user_name": "Dave",
    },
    config_mem,
)
print(f"AI: {result['messages'][-1].content}")
print(f"状态: user_name={result.get('user_name')}, favorite_color={result.get('favorite_color')}")
print()

# 再查询
print("--- 查询用户信息 ---")
result = agent_memory_tool.invoke(
    {"messages": [{"role": "user", "content": "你记得我叫什么吗？我喜欢什么颜色？"}]},
    config_mem,
)
print(f"AI: {result['messages'][-1].content}")
print()
