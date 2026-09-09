"""
Demo 16: 长期记忆 (Long-term Memory)
======================================
LangChain 智能体使用 LangGraph 持久化实现长期记忆。

核心知识点：
1. 内存存储 (Memory storage): InMemoryStore，每条记忆组织在 namespace 和 key 下
2. 在工具中读取长期记忆: 通过 runtime.store.get() 读取
3. 从工具中写入长期记忆: 通过 runtime.store.put() 写入
4. 支持跨命名空间搜索（基于内容过滤和向量相似度）

参考文档: https://langchain-doc.cn/v1/python/langchain/long-term-memory.html
"""

from dataclasses import dataclass
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
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
# 定义运行时上下文
# ============================================================
@dataclass
class Context:
    """运行时上下文模式。"""
    user_id: str


# ============================================================
# 第一部分：内存存储 (Memory Storage)
# ============================================================
# InMemoryStore 将数据保存到内存字典中
# 每条记忆组织在 namespace（类似文件夹）和 key（类似文件名）下

def demo_memory_storage():
    """演示内存存储的基本操作。"""
    print("=" * 60)
    print("第一部分：内存存储 (Memory Storage)")
    print("=" * 60)

    store = InMemoryStore()

    user_id = "my-user"
    application_context = "chitchat"
    namespace = (user_id, application_context)

    # 写入记忆
    store.put(
        namespace,
        "a-memory",
        {
            "rules": [
                "用户喜欢简洁直接的语言",
                "用户使用中文交流",
            ],
            "my-key": "my-value",
        },
    )
    print("写入记忆到 namespace:", namespace)

    # 通过 ID 获取记忆
    item = store.get(namespace, "a-memory")
    print(f"\n通过 ID 获取: {item.value}")

    # 搜索记忆
    items = store.search(
        namespace,
        filter={"my-key": "my-value"},
        query="language preferences"
    )
    print(f"\n搜索结果: {len(items)} 条匹配")
    for item in items:
        print(f"  - Key: {item.key}, Value: {item.value}")

    print()


# ============================================================
# 第二部分：在工具中读取长期记忆
# ============================================================

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """获取用户信息。"""
    store = runtime.store
    user_id = runtime.context.user_id

    user_info = store.get(("users",), user_id)
    if user_info:
        return f"用户信息: {user_info.value}"
    return "未知用户"


@tool
def get_user_preferences(runtime: ToolRuntime[Context]) -> str:
    """获取用户偏好设置。"""
    store = runtime.store
    user_id = runtime.context.user_id

    prefs = store.get(("preferences",), user_id)
    if prefs:
        return f"用户偏好: {prefs.value}"
    return "未找到用户偏好"


def demo_read_from_store():
    """演示在工具中读取长期记忆。"""
    print("=" * 60)
    print("第二部分：在工具中读取长期记忆")
    print("=" * 60)

    store = InMemoryStore()
    store.put(
        ("users",),
        "user_123",
        {"name": "张三", "language": "中文", "plan": "premium"}
    )
    store.put(
        ("preferences",),
        "user_123",
        {"communication_style": "简洁直接", "timezone": "Asia/Shanghai"}
    )

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[get_user_info, get_user_preferences],
        store=store,
        context_schema=Context,
        system_prompt="你是一位乐于助人的助手，可以使用工具获取用户信息。",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请帮我查看我的用户信息和偏好"}]},
        context=Context(user_id="user_123")
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第三部分：从工具中写入长期记忆
# ============================================================

class UserInfo(TypedDict):
    """用户信息结构定义。"""
    name: str
    email: str


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """保存用户信息到长期记忆。"""
    store = runtime.store
    user_id = runtime.context.user_id

    store.put(("users",), user_id, user_info)
    return "用户信息已保存。"


@tool
def save_preference(
    key: str,
    value: str,
    runtime: ToolRuntime[Context]
) -> str:
    """保存用户偏好设置到长期记忆。"""
    store = runtime.store
    user_id = runtime.context.user_id

    existing = store.get(("preferences",), user_id)
    if existing:
        prefs = existing.value
    else:
        prefs = {}

    prefs[key] = value
    store.put(("preferences",), user_id, prefs)
    return f"偏好 '{key}' 已保存为 '{value}'"


def demo_write_to_store():
    """演示从工具中写入长期记忆。"""
    print("=" * 60)
    print("第三部分：从工具中写入长期记忆")
    print("=" * 60)

    store = InMemoryStore()

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[save_user_info, save_preference, get_user_info, get_user_preferences],
        store=store,
        context_schema=Context,
        system_prompt="""你是一位乐于助人的助手。
你可以保存和读取用户信息。
当用户提供个人信息时，请主动保存。
当用户询问信息时，请使用工具查找。"""
    )

    # 保存信息
    print("测试1: 保存用户信息")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫李四，我的邮箱是 lisi@example.com"}]},
        context=Context(user_id="user_456")
    )
    print(f"回复: {result['messages'][-1].content}")

    # 读取信息
    print("\n测试2: 读取已保存的信息")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请查看我的用户信息"}]},
        context=Context(user_id="user_456")
    )
    print(f"回复: {result['messages'][-1].content}")

    # 直接访问 store 验证
    print("\n直接访问 store 验证:")
    stored = store.get(("users",), "user_456")
    print(f"  store 中的用户信息: {stored.value if stored else '无'}")
    print()


# ============================================================
# 第四部分：长期记忆的生命周期
# ============================================================

def demo_memory_lifecycle():
    """演示长期记忆在多个会话间的持久性。"""
    print("=" * 60)
    print("第四部分：长期记忆的生命周期（跨会话持久化）")
    print("=" * 60)

    # 注意: InMemoryStore 在内存中，进程结束后数据丢失
    # 生产环境应使用基于数据库的存储（如 PostgresStore）

    store = InMemoryStore()
    store.put(("memories",), "user_789", {
        "facts": [
            "用户是软件工程师",
            "用户正在学习 LangChain",
            "用户偏好中文交流"
        ]
    })

    # 会话1: 保存信息
    print("会话1: 保存用户信息")
    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[save_user_info, get_user_info],
        store=store,
        context_schema=Context,
        system_prompt="你是一位助手，可以保存和读取用户信息。",
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "我是王五，邮箱 wangwu@example.com"}]},
        context=Context(user_id="user_789")
    )
    print("  信息已保存")

    # 会话2: 读取信息（模拟新会话）
    print("\n会话2: 读取之前保存的信息（模拟新会话）")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请告诉我我的用户信息"}]},
        context=Context(user_id="user_789")
    )
    print(f"  回复: {result['messages'][-1].content}")

    # 直接验证 store
    print("\n验证 store 中的数据:")
    user_data = store.get(("users",), "user_789")
    if user_data:
        print(f"  用户数据: {user_data.value}")
    memories = store.get(("memories",), "user_789")
    if memories:
        print(f"  记忆数据: {memories.value}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("Demo 16: 长期记忆 (Long-term Memory)\n")
    print("LangChain 智能体使用 LangGraph 持久化实现长期记忆。")
    print("长期记忆存储在 Store 中，组织在 namespace 和 key 下。\n")

    # 演示内存存储
    demo_memory_storage()

    # 演示读取长期记忆
    demo_read_from_store()

    # 演示写入长期记忆
    demo_write_to_store()

    # 演示长期记忆生命周期
    demo_memory_lifecycle()

    print("\n长期记忆总结:")
    print("1. InMemoryStore: 内存存储，生产环境用数据库存储")
    print("2. namespace: 用于组织记忆（类似文件夹），通常包含用户/组织 ID")
    print("3. key: 命名空间内的唯一标识（类似文件名）")
    print("4. 读取: 通过 runtime.store.get(namespace, key)")
    print("5. 写入: 通过 runtime.store.put(namespace, key, value)")
    print("6. 搜索: 通过 store.search() 支持内容过滤和向量相似度")
    print("7. 长期记忆跨会话持久化，不同于短期记忆（checkpointer）")
