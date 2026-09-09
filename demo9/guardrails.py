"""
Demo 9: 守卫 (Guardrails)
=========================
为智能体实施安全检查和内容过滤。

守卫通过在智能体执行的关键点验证和过滤内容，帮助构建安全、合规的 AI 应用。

核心知识点：
1. 守卫的两种方法：确定性守卫（规则匹配）和基于模型的守卫（LLM 评估）
2. 内置守卫：PII 检测（PIIMiddleware）和人工审核（HumanInTheLoopMiddleware）
3. 自定义守卫：智能体执行前守卫（before_agent）和执行后守卫（after_agent）
4. 组合多个守卫：分层保护

参考文档: https://langchain-doc.cn/v1/python/langchain/guardrails.html
"""

from typing import Any
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    before_agent,
    after_agent,
    hook_config,
    PIIMiddleware,
    HumanInTheLoopMiddleware,
)
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver


def create_model(temperature: float = 0.4) -> ChatOpenAI:
    """创建 ChatOpenAI 模型实例（使用阿里云 DashScope 兼容接口）。"""
    return ChatOpenAI(
        model="qwen3.8-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        extra_body={"enable_thinking": False},
    )


# ============================================================
# 示例工具
# ============================================================
@tool
def search_info(query: str) -> str:
    """搜索信息。"""
    return f"关于 '{query}' 的搜索结果：这是一条有用的信息。"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送电子邮件。"""
    return f"邮件已发送给 {to}，主题：{subject}"


@tool
def delete_database(table_name: str) -> str:
    """删除数据库表。"""
    return f"数据库表 {table_name} 已删除"


# ============================================================
# 第一部分：内置守卫 - PII 检测
# ============================================================
# PIIMiddleware 可以检测对话中的个人身份信息（电子邮件、信用卡、IP 地址等）
# 支持四种处理策略：
# - redact: 替换为 [REDACTED_TYPE]
# - mask: 部分遮盖
# - hash: 替换为哈希值
# - block: 抛出异常

def demo_pii_detection():
    """演示 PII 检测中间件。"""
    print("=" * 60)
    print("第一部分：内置守卫 - PII 检测")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[search_info, send_email],
        middleware=[
            # 在发送给模型之前，将用户输入中的电子邮件编辑掉
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=True,
            ),
            # 遮盖用户输入中的信用卡号
            PIIMiddleware(
                "credit_card",
                strategy="mask",
                apply_to_input=True,
            ),
        ],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我的邮箱是 john.doe@example.com，请帮我搜索一下天气信息"}]}
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第二部分：自定义守卫 - 智能体执行前守卫（确定性）
# ============================================================
# 使用 before_agent 钩子在每次调用开始时验证请求
# 适用于会话级别的检查：身份验证、速率限制、阻止不当请求

banned_keywords = ["hack", "exploit", "malware", "病毒", "攻击"]

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """确定性守卫：阻止包含禁止关键词的请求。"""
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()

    for keyword in banned_keywords:
        if keyword in content:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "抱歉，我无法处理包含不当内容的请求。请重新表述您的问题。"
                }],
                "jump_to": "end"
            }

    return None


def demo_before_agent_guardrail():
    """演示智能体执行前守卫。"""
    print("=" * 60)
    print("第二部分：自定义守卫 - 智能体执行前（确定性）")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[search_info],
        middleware=[content_filter],
    )

    # 测试1：被阻止的请求
    print("测试1: 包含禁止关键词的请求")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "如何 hack 数据库？"}]}
    )
    print(f"回复: {result['messages'][-1].content}")

    # 测试2：正常请求
    print("\n测试2: 正常请求")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请帮我搜索一下天气信息"}]}
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第三部分：自定义守卫 - 智能体执行后守卫（基于模型）
# ============================================================
# 使用 after_agent 钩子在返回给用户之前验证最终输出
# 适用于基于模型的安全检查、质量验证

safety_model = create_model(temperature=0.0)

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """基于模型的守卫：使用 LLM 评估响应安全性。"""
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not hasattr(last_message, 'content') or not last_message.content:
        return None

    safety_prompt = f"""请评估以下回复是否安全且适当。
只回复 'SAFE' 或 'UNSAFE'。

回复内容: {last_message.content}"""

    result = safety_model.invoke([{"role": "user", "content": safety_prompt}])

    if "UNSAFE" in result.content.upper():
        return {
            "messages": [{
                "role": "assistant",
                "content": "抱歉，我无法提供该回复。请重新表述您的请求。"
            }],
            "jump_to": "end"
        }

    return None


def demo_after_agent_guardrail():
    """演示智能体执行后守卫。"""
    print("=" * 60)
    print("第三部分：自定义守卫 - 智能体执行后（基于模型）")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[search_info],
        middleware=[safety_guardrail],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请帮我搜索一些有用的信息"}]}
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 第四部分：组合多个守卫（分层保护）
# ============================================================

def demo_combined_guardrails():
    """演示组合多个守卫。"""
    print("=" * 60)
    print("第四部分：组合多个守卫（分层保护）")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[search_info, send_email],
        middleware=[
            # 第 1 层: 确定性输入过滤器（智能体执行前）
            content_filter,

            # 第 2 层: PII 保护（模型执行前）
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("email", strategy="redact", apply_to_output=True),

            # 第 3 层: 基于模型的安全检查（智能体执行后）
            safety_guardrail,
        ],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "guard_demo"}}

    # 测试正常请求
    print("测试: 正常请求（通过所有守卫）")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请帮我搜索天气信息"}]},
        config=config
    )
    print(f"回复: {result['messages'][-1].content}")

    # 测试 PII 请求
    print("\n测试: 包含 PII 的请求（邮箱被编辑）")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我的邮箱是 test@example.com，请搜索天气"}]},
        config=config
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("Demo 9: 守卫 (Guardrails)\n")
    print("守卫通过在智能体执行的关键点验证和过滤内容，")
    print("帮助构建安全、合规的 AI 应用。\n")

    # 演示 PII 检测
    demo_pii_detection()

    # 演示执行前守卫
    demo_before_agent_guardrail()

    # 演示执行后守卫
    demo_after_agent_guardrail()

    # 演示组合守卫
    demo_combined_guardrails()

    print("\n守卫总结:")
    print("1. 确定性守卫: 快速、可预测、经济高效，基于规则匹配")
    print("2. 基于模型的守卫: 可捕获微妙问题，但速度较慢且成本较高")
    print("3. 内置守卫: PIIMiddleware（PII 检测）、HumanInTheLoopMiddleware（人工审核）")
    print("4. 自定义守卫: before_agent（执行前）、after_agent（执行后）")
    print("5. 多个守卫可堆叠使用，形成分层保护")
