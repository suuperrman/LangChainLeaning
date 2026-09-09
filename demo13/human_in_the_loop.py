"""
Demo 13: 人机交互 (Human-in-the-Loop)
=======================================
HITL 中间件允许在代理工具调用中添加人工监督。

核心知识点：
1. 三种中断决策类型：approve（批准）、edit（编辑）、reject（拒绝）
2. 配置中断：通过 interrupt_on 映射工具到允许的决策类型
3. 执行生命周期：模型生成响应 -> 中间件检查工具调用 -> 触发中断 -> 等待人工决策
4. 响应中断：使用 Command(resume={...}) 恢复执行
5. 需要 checkpointer 来持久化中断时的图状态

参考文档: https://langchain-doc.cn/v1/python/langchain/human-in-the-loop.html
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


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
# 示例工具
# ============================================================
@tool
def write_file(filename: str, content: str) -> str:
    """写入文件到磁盘。"""
    return f"文件 {filename} 已写入，内容: {content[:50]}..."


@tool
def execute_sql(query: str) -> str:
    """执行 SQL 查询。"""
    return f"SQL 执行成功: {query[:50]}..."


@tool
def read_data(source: str) -> str:
    """读取数据（安全操作，无需批准）。"""
    return f"从 {source} 读取到数据"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送电子邮件。"""
    return f"邮件已发送给 {to}，主题: {subject}"


# ============================================================
# 第一部分：配置人机交互
# ============================================================

def create_hitl_agent():
    """创建带有人机交互的智能体。"""
    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[write_file, execute_sql, read_data, send_email],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # 写文件: 允许所有决策（批准、编辑、拒绝）
                    "write_file": True,
                    # 执行 SQL: 只允许批准和拒绝，不允许编辑
                    "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                    # 发送邮件: 允许所有决策
                    "send_email": True,
                    # 读取数据: 安全操作，无需批准
                    "read_data": False,
                },
                description_prefix="工具执行等待批准",
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    return agent


# ============================================================
# 第二部分：批准 (Approve) 决策
# ============================================================

def demo_approve():
    """演示批准决策。"""
    print("=" * 60)
    print("第二部分：批准 (Approve) 决策")
    print("=" * 60)

    agent = create_hitl_agent()
    config = {"configurable": {"thread_id": "approve_demo"}}

    # 第一次调用：智能体会请求执行 SQL，然后中断等待批准
    print("1. 请求删除旧记录:")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请执行 SQL 删除旧记录: DELETE FROM records WHERE created_at < '2024-01-01'"}]},
        config=config
    )

    if '__interrupt__' in result:
        print(f"   中断! 等待人工审核...")
        interrupt_data = result['__interrupt__']
        print(f"   中断数据: {interrupt_data}")

        # 批准操作
        print("\n2. 批准执行:")
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )
        print(f"   最终回复: {result['messages'][-1].content}")
    else:
        print(f"   直接回复: {result['messages'][-1].content}")
    print()


# ============================================================
# 第三部分：编辑 (Edit) 决策
# ============================================================

def demo_edit():
    """演示编辑决策。"""
    print("=" * 60)
    print("第三部分：编辑 (Edit) 决策")
    print("=" * 60)

    agent = create_hitl_agent()
    config = {"configurable": {"thread_id": "edit_demo"}}

    print("1. 请求写入文件:")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请将 'hello world' 写入 test.txt 文件"}]},
        config=config
    )

    if '__interrupt__' in result:
        print(f"   中断! 等待人工审核...")
        interrupt_data = result['__interrupt__']
        print(f"   中断数据: {interrupt_data}")

        # 编辑操作：修改文件名
        print("\n2. 编辑操作（修改文件名）:")
        result = agent.invoke(
            Command(resume={
                "decisions": [{
                    "type": "edit",
                    "edited_action": {
                        "name": "write_file",
                        "args": {
                            "filename": "edited_test.txt",
                            "content": "hello world"
                        }
                    }
                }]
            }),
            config=config
        )
        print(f"   最终回复: {result['messages'][-1].content}")
    else:
        print(f"   直接回复: {result['messages'][-1].content}")
    print()


# ============================================================
# 第四部分：拒绝 (Reject) 决策
# ============================================================

def demo_reject():
    """演示拒绝决策。"""
    print("=" * 60)
    print("第四部分：拒绝 (Reject) 决策")
    print("=" * 60)

    agent = create_hitl_agent()
    config = {"configurable": {"thread_id": "reject_demo"}}

    print("1. 请求发送邮件:")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请给 team@example.com 发送邮件，主题是测试，内容是这是测试邮件"}]},
        config=config
    )

    if '__interrupt__' in result:
        print(f"   中断! 等待人工审核...")
        interrupt_data = result['__interrupt__']
        print(f"   中断数据: {interrupt_data}")

        # 拒绝操作
        print("\n2. 拒绝操作:")
        result = agent.invoke(
            Command(resume={
                "decisions": [{
                    "type": "reject",
                    "message": "不，这个操作不合适，请取消发送。"
                }]
            }),
            config=config
        )
        print(f"   最终回复: {result['messages'][-1].content}")
    else:
        print(f"   直接回复: {result['messages'][-1].content}")
    print()


# ============================================================
# 第五部分：安全操作（无需批准）
# ============================================================

def demo_safe_operation():
    """演示安全操作（无需人工审核）。"""
    print("=" * 60)
    print("第五部分：安全操作（无需人工审核）")
    print("=" * 60)

    agent = create_hitl_agent()
    config = {"configurable": {"thread_id": "safe_demo"}}

    print("请求读取数据（安全操作，无需批准）:")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请从数据库读取用户数据"}]},
        config=config
    )
    print(f"回复: {result['messages'][-1].content}\n")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("Demo 13: 人机交互 (Human-in-the-Loop)\n")
    print("HITL 中间件允许在代理工具调用中添加人工监督。")
    print("当模型提出需要人工审查的操作时，中间件会暂停执行并等待决策。\n")

    # 演示安全操作
    demo_safe_operation()

    # 演示批准决策
    demo_approve()

    # 演示编辑决策
    demo_edit()

    # 演示拒绝决策
    demo_reject()

    print("\n人机交互总结:")
    print("1. 三种决策类型:")
    print("   - approve: 按原样批准并执行")
    print("   - edit: 修改后执行")
    print("   - reject: 拒绝并添加反馈")
    print("2. 必须配置 checkpointer 来持久化中断时的状态")
    print("3. 使用 Command(resume={...}) 恢复暂停的对话")
    print("4. 通过 interrupt_on 配置哪些工具需要人工审核")
    print("5. 执行生命周期: 模型响应 -> 检查工具调用 -> 中断 -> 人工决策 -> 恢复")
