"""
Demo 11: 智能体中的上下文工程 (Context Engineering)
====================================================
上下文工程就是以正确的格式提供正确的信息和工具，以便 LLM 能够完成任务。

核心知识点：
1. 上下文的三大类型：模型上下文（瞬态）、工具上下文（持久性）、生命周期上下文（持久性）
2. 数据源：运行时上下文（静态配置）、状态（短期记忆）、存储（长期记忆）
3. 模型上下文控制：系统提示、消息、工具、模型、响应格式
4. 中间件是实现上下文工程的底层机制（挂接到代理生命周期中的任何步骤）

参考文档: https://langchain-doc.cn/v1/python/langchain/context-engineering.html
"""

from dataclasses import dataclass
from typing import Any, Callable

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    before_model,
)
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore


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
# 定义上下文模式
# ============================================================
@dataclass
class Context:
    """运行时上下文模式。"""
    user_id: str
    user_name: str
    user_role: str = "user"
    deployment_env: str = "development"


# ============================================================
# 示例工具
# ============================================================
@tool
def public_search(query: str) -> str:
    """公开搜索工具，所有用户可用。"""
    return f"公开搜索结果: {query}"


@tool
def private_search(query: str) -> str:
    """私有搜索工具，仅认证用户可用。"""
    return f"私有搜索结果: {query}"


@tool
def advanced_search(query: str) -> str:
    """高级搜索工具，仅管理员可用。"""
    return f"高级搜索结果: {query}"


@tool
def save_data(data: str) -> str:
    """保存数据。"""
    return f"数据已保存: {data}"


# ============================================================
# 第一部分：基于状态的动态系统提示
# ============================================================
# 从状态中访问消息计数或会话上下文

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    """根据对话状态动态生成系统提示。"""
    message_count = len(request.messages)
    base = "你是一位乐于助人的助手。"

    if message_count > 5:
        base += "\n这是一个较长的对话，请保持简洁。"

    return base


def demo_state_based_prompt():
    """演示基于状态的动态系统提示。"""
    print("=" * 60)
    print("第一部分：基于状态的动态系统提示")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[public_search],
        middleware=[state_aware_prompt],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]}
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第二部分：基于存储的动态系统提示
# ============================================================
# 从长期记忆中访问用户偏好

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    """从长期记忆中访问用户偏好，动态生成系统提示。"""
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    base = "你是一位乐于助人的助手。"

    if store:
        user_prefs = store.get(("preferences",), user_id)
        if user_prefs:
            style = user_prefs.value.get("communication_style", "balanced")
            base += f"\n用户偏好 {style} 的回复风格。"

    return base


def demo_store_based_prompt():
    """演示基于存储的动态系统提示。"""
    print("=" * 60)
    print("第二部分：基于存储的动态系统提示")
    print("=" * 60)

    store = InMemoryStore()
    store.put(
        ("preferences",),
        "1",
        {"communication_style": "简洁直接", "language": "中文"}
    )

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[public_search],
        middleware=[store_aware_prompt],
        context_schema=Context,
        store=store,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请帮我搜索一些信息"}]},
        context=Context(user_id="1", user_name="张三")
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第三部分：基于运行时上下文的动态系统提示
# ============================================================
# 从运行时上下文中访问用户角色和部署环境

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """根据运行时上下文动态生成系统提示。"""
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "你是一位乐于助人的助手。"

    if user_role == "admin":
        base += "\n用户具有管理员权限，可以执行所有操作。"
    else:
        base += "\n用户具有只读权限，请引导用户使用只读操作。"

    if env == "production":
        base += "\n生产环境，请谨慎处理任何数据修改。"

    return base


def demo_runtime_context_prompt():
    """演示基于运行时上下文的动态系统提示。"""
    print("=" * 60)
    print("第三部分：基于运行时上下文的动态系统提示")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[public_search],
        middleware=[context_aware_prompt],
        context_schema=Context,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，我可以做什么操作？"}]},
        context=Context(user_id="1", user_name="张三", user_role="admin", deployment_env="production")
    )
    print(f"管理员回复: {result['messages'][-1].content}\n")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，我可以做什么操作？"}]},
        context=Context(user_id="2", user_name="李四", user_role="user", deployment_env="development")
    )
    print(f"普通用户回复: {result['messages'][-1].content}\n")


# ============================================================
# 第四部分：动态工具选择
# ============================================================
# 根据运行时上下文中的用户权限过滤工具

@wrap_model_call
def role_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据用户角色过滤可用工具。"""
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        pass
    elif user_role == "editor":
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)
    else:
        tools = [t for t in request.tools if t.name.startswith("public")]
        request = request.override(tools=tools)

    return handler(request)


def demo_dynamic_tool_selection():
    """演示动态工具选择。"""
    print("=" * 60)
    print("第四部分：动态工具选择（基于运行时上下文）")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[public_search, private_search, advanced_search],
        middleware=[role_based_tools],
        context_schema=Context,
        system_prompt="你是一位助手，可以使用搜索工具帮助用户。",
    )

    # 管理员用户
    print("管理员用户（可使用所有工具）:")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请使用高级搜索查询天气"}]},
        context=Context(user_id="1", user_name="张三", user_role="admin")
    )
    print(f"回复: {result['messages'][-1].content}\n")

    # 普通用户
    print("普通用户（只能使用公开工具）:")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请搜索天气信息"}]},
        context=Context(user_id="2", user_name="李四", user_role="user")
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第五部分：消息注入（瞬态更新）
# ============================================================
# 使用 wrap_model_call 进行瞬态更新——修改发送给模型的消息，不更改状态

@wrap_model_call
def inject_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """在最新消息之前注入上下文信息。"""
    user_name = request.runtime.context.user_name

    context_msg = f"当前用户: {user_name}，请根据用户信息进行回复。"
    messages = [
        *request.messages,
        {"role": "user", "content": context_msg}
    ]
    request = request.override(messages=messages)

    return handler(request)


def demo_message_injection():
    """演示消息注入。"""
    print("=" * 60)
    print("第五部分：消息注入（瞬态更新）")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[public_search],
        middleware=[inject_context],
        context_schema=Context,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，你知道我是谁吗？"}]},
        context=Context(user_id="1", user_name="张三", user_role="user")
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("Demo 11: 智能体中的上下文工程 (Context Engineering)\n")
    print("上下文工程就是以正确的格式提供正确的信息和工具，")
    print("以便 LLM 能够完成任务。这是 AI 工程师的核心工作。\n")

    demo_state_based_prompt()
    demo_store_based_prompt()
    demo_runtime_context_prompt()
    demo_dynamic_tool_selection()
    demo_message_injection()

    print("\n上下文工程总结:")
    print("1. 模型上下文(瞬态): 系统提示、消息、工具、模型、响应格式")
    print("2. 工具上下文(持久性): 工具可访问状态、存储、运行时上下文")
    print("3. 生命周期上下文(持久性): 模型调用和工具调用之间的内容")
    print("4. 数据源: 运行时上下文(静态配置)、状态(短期记忆)、存储(长期记忆)")
    print("5. 中间件是实现上下文工程的底层机制")
