"""
Demo 14: 多智能体系统 (Multi-Agent)
=====================================
多智能体系统将复杂应用拆分成多个协同工作的专业化智能体。

核心知识点：
1. 两种多智能体模式：
   - 工具调用 (Tool Calling): 主管智能体将其他智能体作为工具调用（集中式控制）
   - 交接 (Handoffs): 智能体直接转移控制权（分散式控制）
2. 上下文工程：控制每个智能体看到哪些信息
3. 控制子智能体的输入和输出
4. 选择模式：根据需求选择工具调用或交接

参考文档: https://langchain-doc.cn/v1/python/langchain/multi-agent.html
"""

from typing import Annotated
from dataclasses import dataclass

from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_core.messages import ToolMessage


# ============================================================
# 统一的模型创建方式（与 demo2 保持一致，使用阿里云 DashScope 兼容接口）
# ============================================================
def create_model(temperature: float = 0.4) -> ChatOpenAI:
    """创建 ChatOpenAI 模型实例（使用阿里云 DashScope 兼容接口）。"""
    return ChatOpenAI(
        model="qwen3.8-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        extra_body={"enable_thinking": False},
    )


# ============================================================
# 第一部分：工具调用模式（Tool Calling）
# ============================================================
# 主管（supervisor）智能体将其他智能体作为工具调用
# 控制流是集中式的：所有路由都通过主管智能体

# --- 子智能体 1：研究助手 ---
research_agent = create_agent(
    model=create_model(temperature=0.3),
    tools=[],
    system_prompt="你是一位专业的研究助手。请提供详细、准确的信息。",
    name="research_agent",
)


@tool(
    "research_assistant",
    description="当需要查找信息、分析数据或进行研究时调用此工具。传入需要研究的问题。"
)
def call_research(query: str) -> str:
    """调用研究助手进行信息查找和分析。"""
    result = research_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


# --- 子智能体 2：写作助手 ---
writing_agent = create_agent(
    model=create_model(temperature=0.5),
    tools=[],
    system_prompt="你是一位专业的写作助手。请根据要求撰写或优化文本内容。",
    name="writing_agent",
)


@tool(
    "writing_assistant",
    description="当需要撰写文档、邮件、报告或优化文本时调用此工具。传入写作要求。"
)
def call_writing(task: str) -> str:
    """调用写作助手撰写或优化文本。"""
    result = writing_agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )
    return result["messages"][-1].content


# --- 子智能体 3：翻译助手 ---
translation_agent = create_agent(
    model=create_model(temperature=0.3),
    tools=[],
    system_prompt="你是一位专业的翻译助手。可以在中文和英文之间进行翻译。",
    name="translation_agent",
)


@tool(
    "translation_assistant",
    description="当需要翻译文本时调用此工具。传入需要翻译的内容和目标语言。"
)
def call_translation(task: str) -> str:
    """调用翻译助手进行翻译。"""
    result = translation_agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )
    return result["messages"][-1].content


def demo_tool_calling():
    """演示工具调用模式。"""
    print("=" * 60)
    print("第一部分：工具调用模式（Tool Calling）")
    print("=" * 60)

    # 主管智能体（控制器）
    supervisor = create_agent(
        model=create_model(temperature=0.3),
        tools=[call_research, call_writing, call_translation],
        system_prompt="""你是一个任务编排智能体（主管）。
你可以调用以下专业子智能体来完成任务：
- research_assistant: 查找信息、分析数据
- writing_assistant: 撰写文档、优化文本
- translation_assistant: 翻译文本

根据用户需求，选择合适的子智能体来执行任务。
将子智能体的结果整理后返回给用户。"""
    )

    print("测试1: 需要研究+写作的任务")
    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": "请研究一下人工智能的发展历史，然后写一份简短的报告"}]}
    )
    print(f"回复: {result['messages'][-1].content}\n")

    print("测试2: 需要翻译的任务")
    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": "请把 '机器学习是人工智能的一个分支' 翻译成英文"}]}
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第二部分：控制对子智能体的输入
# ============================================================
# 通过 ToolRuntime 访问主智能体的状态，自定义传给子智能体的输入

class CustomState(AgentState):
    """自定义状态，可以包含额外的字段。"""
    pass


@tool(
    "context_aware_research",
    description="带有上下文的研究助手。会根据对话历史提供更精准的研究结果。"
)
def call_research_with_context(
    query: str,
    runtime: ToolRuntime[None, CustomState]
) -> str:
    """根据对话上下文调用研究助手。"""
    messages = runtime.state["messages"]

    context = ""
    if len(messages) > 1:
        recent_messages = messages[-3:]
        context = "对话历史:\n"
        for msg in recent_messages:
            context += f"- {msg.type}: {msg.content[:100]}\n"

    full_query = f"{context}\n当前问题: {query}"
    result = research_agent.invoke(
        {"messages": [{"role": "user", "content": full_query}]}
    )
    return result["messages"][-1].content


def demo_control_input():
    """演示控制子智能体的输入。"""
    print("=" * 60)
    print("第二部分：控制对子智能体的输入")
    print("=" * 60)

    supervisor = create_agent(
        model=create_model(temperature=0.3),
        tools=[call_research_with_context],
        system_prompt="你是一个任务编排智能体，可以使用研究助手来帮助用户。",
        state_schema=CustomState,
    )

    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": "请研究一下 Python 和 Java 的主要区别"}]}
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第三部分：控制来自子智能体的输出
# ============================================================
# 使用 Command 对象返回自定义状态和消息

@tool(
    "research_with_metadata",
    description="研究助手，返回结果和元数据。"
)
def call_research_with_output(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """调用研究助手并返回带元数据的结果。"""
    result = research_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    return Command(update={
        "messages": [
            ToolMessage(
                content=f"[研究结果] {result['messages'][-1].content}",
                tool_call_id=tool_call_id,
            )
        ]
    })


def demo_control_output():
    """演示控制来自子智能体的输出。"""
    print("=" * 60)
    print("第三部分：控制来自子智能体的输出")
    print("=" * 60)

    supervisor = create_agent(
        model=create_model(temperature=0.3),
        tools=[call_research_with_output],
        system_prompt="你是一个任务编排智能体，使用研究助手获取信息。",
    )

    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": "请研究一下什么是深度学习"}]}
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第四部分：多智能体模式对比
# ============================================================

def demo_pattern_comparison():
    """对比两种多智能体模式。"""
    print("=" * 60)
    print("第四部分：多智能体模式对比")
    print("=" * 60)

    print("""
两种模式对比:

| 模式     | 工作原理                          | 控制流     | 适用场景                 |
|----------|-----------------------------------|------------|--------------------------|
| 工具调用 | 主管智能体将子智能体作为工具调用    | 集中式     | 任务编排、结构化工作流    |
| 交接     | 当前智能体将控制权转移给另一个智能体 | 分散式     | 多领域对话、专家接管      |

选择建议:
- 需要集中式控制？          -> 工具调用
- 希望子智能体直接与用户交互？ -> 交接
- 需要复杂的专家间对话？     -> 交接
- 可以混合使用两种模式
""")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("Demo 14: 多智能体系统 (Multi-Agent)\n")
    print("多智能体系统将复杂应用拆分成多个协同工作的专业化智能体。")
    print("当单个智能体拥有太多工具或上下文过大时，多智能体架构非常有用。\n")

    # 演示工具调用模式
    demo_tool_calling()

    # 演示控制输入
    demo_control_input()

    # 演示控制输出
    demo_control_output()

    # 模式对比
    demo_pattern_comparison()

    print("\n多智能体系统总结:")
    print("1. 工具调用模式: 主管集中控制，子智能体返回结果给主管")
    print("2. 交接模式: 智能体分散控制，用户直接与新智能体交互")
    print("3. 控制输入: 通过 ToolRuntime 访问状态，自定义子智能体的输入")
    print("4. 控制输出: 使用 Command 对象返回自定义状态和消息")
    print("5. 上下文工程是核心: 决定每个智能体看到哪些信息")
