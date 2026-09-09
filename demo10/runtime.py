"""
Demo 10: 运行时 (Runtime)
==========================
LangChain 的 create_agent 在 LangGraph 的运行时环境下运行。

Runtime 对象包含：
1. Context (上下文): 静态信息，如用户 ID、数据库连接等
2. Store (存储): BaseStore 实例，用于长期记忆
3. Stream writer (流写入器): 用于 "custom" 流模式的信息流式传输

核心知识点：
1. Runtime 的三大组成部分：Context、Store、Stream writer
2. 在工具内部通过 ToolRuntime 访问运行时
3. 在中间件内部通过 request.runtime 访问运行时
4. 使用 context_schema 定义上下文结构

参考文档: https://langchain-doc.cn/v1/python/langchain/runtime.html
"""

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
    before_model,
    after_model,
)
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
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
# 定义运行时上下文模式
# ============================================================
@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str
    user_name: str
    user_role: str = "user"


# ============================================================
# 第一部分：在工具内部访问运行时
# ============================================================
# 使用 ToolRuntime 参数访问工具内部的 Runtime 对象
# 可以访问：上下文（context）、存储（store）、流写入器（stream writer）

@tool
def fetch_user_info(runtime: ToolRuntime[Context]) -> str:
    """获取当前用户的信息。"""
    user_id = runtime.context.user_id
    user_name = runtime.context.user_name

    info = f"用户 ID: {user_id}, 用户名: {user_name}"

    if runtime.store:
        stored_data = runtime.store.get(("users",), user_id)
        if stored_data:
            info += f", 存储数据: {stored_data.value}"

    return info


@tool
def fetch_user_preferences(runtime: ToolRuntime[Context]) -> str:
    """从长期记忆中获取用户偏好。"""
    user_id = runtime.context.user_id

    if runtime.store:
        memory = runtime.store.get(("preferences",), user_id)
        if memory:
            return f"用户偏好: {memory.value}"
        return "未找到用户偏好"

    return "存储不可用"


@tool
def report_progress(message: str, runtime: ToolRuntime[Context]) -> str:
    """通过流写入器报告工具执行进度。"""
    user_name = runtime.context.user_name
    progress_msg = f"[{user_name}] 进度: {message}"

    if runtime.writer:
        runtime.writer(progress_msg)

    return progress_msg


# ============================================================
# 第二部分：在中间件内部访问运行时
# ============================================================
# 使用 request.runtime 访问中间件装饰器内部的 Runtime 对象

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """根据运行时上下文动态生成系统提示。"""
    user_name = request.runtime.context.user_name
    user_role = request.runtime.context.user_role

    prompt = f"你是一位乐于助人的助手。请称呼用户为 {user_name}。"

    if user_role == "admin":
        prompt += "\n用户具有管理员权限，可以执行所有操作。"
    else:
        prompt += "\n用户具有普通权限，只能执行基本操作。"

    return prompt


@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict[str, Any] | None:
    """在模型调用之前记录日志。"""
    user_name = runtime.context.user_name
    message_count = len(state["messages"])
    print(f"  [日志] 开始处理 {user_name} 的请求，当前消息数: {message_count}")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict[str, Any] | None:
    """在模型调用之后记录日志。"""
    user_name = runtime.context.user_name
    print(f"  [日志] 完成处理 {user_name} 的请求")
    return None


# ============================================================
# 第三部分：综合演示 - 结合工具和中间件
# ============================================================

def demo_runtime_in_tools():
    """演示在工具内部访问运行时。"""
    print("=" * 60)
    print("第一部分：在工具内部访问运行时")
    print("=" * 60)

    # 创建 InMemoryStore 并预存一些数据
    store = InMemoryStore()
    store.put(
        ("users",),
        "1",
        {"name": "张三", "language": "中文", "role": "admin"}
    )
    store.put(
        ("preferences",),
        "1",
        {"communication_style": "简洁直接", "timezone": "Asia/Shanghai"}
    )

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[fetch_user_info, fetch_user_preferences],
        context_schema=Context,
        store=store,
        system_prompt="你是一位乐于助人的助手。",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请帮我查看我的用户信息和偏好"}]},
        context=Context(user_id="1", user_name="张三", user_role="admin")
    )
    print(f"回复: {result['messages'][-1].content}\n")


def demo_runtime_in_middleware():
    """演示在中间件内部访问运行时。"""
    print("=" * 60)
    print("第二部分：在中间件内部访问运行时")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[],
        middleware=[dynamic_system_prompt, log_before_model, log_after_model],
        context_schema=Context,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        context=Context(user_id="2", user_name="李四", user_role="user")
    )
    print(f"回复: {result['messages'][-1].content}\n")


def demo_runtime_with_store():
    """演示运行时与长期记忆存储的结合。"""
    print("=" * 60)
    print("第三部分：运行时与长期记忆存储结合")
    print("=" * 60)

    store = InMemoryStore()
    store.put(
        ("users",),
        "user_123",
        {"name": "王五", "email": "wangwu@example.com", "plan": "premium"}
    )

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[fetch_user_info, fetch_user_preferences, report_progress],
        context_schema=Context,
        store=store,
        system_prompt="你是一位乐于助人的助手，可以使用工具获取用户信息。",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请查看我的账户信息"}]},
        context=Context(user_id="user_123", user_name="王五", user_role="admin")
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("Demo 10: 运行时 (Runtime)\n")
    print("LangChain 的 create_agent 在 LangGraph 的运行时环境下运行。")
    print("Runtime 对象包含 Context、Store 和 Stream writer。\n")

    # 演示在工具内部访问运行时
    demo_runtime_in_tools()

    # 演示在中间件内部访问运行时
    demo_runtime_in_middleware()

    # 演示运行时与长期记忆存储的结合
    demo_runtime_with_store()

    print("\n运行时总结:")
    print("1. Context: 静态信息（用户 ID、数据库连接、权限等），通过 context_schema 定义")
    print("2. Store: BaseStore 实例，用于长期记忆（跨会话持久化）")
    print("3. Stream writer: 通过 'custom' 流模式进行信息流式传输")
    print("4. 在工具中使用 ToolRuntime[Context] 参数访问运行时")
    print("5. 在中间件中使用 request.runtime 访问运行时")
