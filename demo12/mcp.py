"""
Demo 12: 模型上下文协议 (MCP)
===============================
MCP (Model Context Protocol) 是一种开放协议，标准化了应用程序如何向 LLM 提供工具和上下文。

核心知识点：
1. MCP 传输类型：stdio（本地子进程）、Streamable HTTP（远程）、SSE（实时流）
2. 使用 langchain-mcp-adapters 库连接 MCP 服务器
3. MultiServerMCPClient：无状态工具调用
4. 自定义 MCP 服务器：使用 mcp 库的 FastMCP 创建
5. 有状态的工具使用：使用 client.session() 创建持久会话

注意：运行此 demo 需要安装 langchain-mcp-adapters 和 mcp 库
  pip install langchain-mcp-adapters mcp

参考文档: https://langchain-doc.cn/v1/python/langchain/mcp.html
"""

import asyncio
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def create_model(temperature: float = 0.4) -> ChatOpenAI:
    """创建 ChatOpenAI 模型实例（使用阿里云 DashScope 兼容接口）。"""
    return ChatOpenAI(
        model="qwen3.8-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        extra_body={"enable_thinking": False},
    )


# ============================================================
# 第一部分：自定义 MCP 服务器（stdio 传输）
# ============================================================
# 将以下代码保存为单独文件 math_server.py 并运行
# 使用 FastMCP 创建数学运算服务器

MATH_SERVER_CODE = """
# math_server.py - 数学 MCP 服务器（stdio 传输）
# 运行: python math_server.py

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers\"\"\"
    return a * b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    \"\"\"Subtract two numbers\"\"\"
    return a - b

if __name__ == "__main__":
    mcp.run(transport="stdio")
"""

WEATHER_SERVER_CODE = """
# weather_server.py - 天气 MCP 服务器（Streamable HTTP 传输）
# 运行: python weather_server.py
# 启动后在 localhost:8000/mcp 提供服务

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    \"\"\"Get weather for location.\"\"\"
    return f"{location}今天天气晴朗，气温 25 度。"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
"""


# ============================================================
# 第二部分：使用 MCP 工具（无状态模式）
# ============================================================
# MultiServerMCPClient 默认是无状态的
# 每次工具调用创建新的 ClientSession，执行工具，然后清理

async def demo_mcp_tools():
    """演示使用 MCP 工具。"""
    print("=" * 60)
    print("第二部分：使用 MCP 工具（无状态模式）")
    print("=" * 60)

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "math": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["math_server.py"],
                },
                "weather": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/mcp",
                }
            }
        )

        tools = await client.get_tools()
        print(f"获取到 {len(tools)} 个 MCP 工具:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        agent = create_agent(
            model=create_model(temperature=0.3),
            tools=tools,
        )

        math_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "(3 + 5) x 12 等于多少？"}]}
        )
        print(f"数学回答: {math_response['messages'][-1].content}")

        weather_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "北京天气怎么样？"}]}
        )
        print(f"天气回答: {weather_response['messages'][-1].content}\n")

    except ImportError:
        print("请先安装依赖: pip install langchain-mcp-adapters mcp")
    except Exception as e:
        print(f"运行失败（需要先启动 MCP 服务器）: {e}\n")


# ============================================================
# 第三部分：有状态的工具使用
# ============================================================
# 对于需要在工具调用之间维护上下文的有状态服务器
# 使用 client.session() 创建持久化的 ClientSession

async def demo_stateful_mcp():
    """演示有状态的 MCP 工具使用。"""
    print("=" * 60)
    print("第三部分：有状态的 MCP 工具使用")
    print("=" * 60)

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        client = MultiServerMCPClient(
            {
                "math": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["math_server.py"],
                }
            }
        )

        async with client.session("math") as session:
            tools = await load_mcp_tools(session)
            print(f"加载了 {len(tools)} 个工具（持久会话）")

            agent = create_agent(
                model=create_model(temperature=0.3),
                tools=tools,
            )

            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "3 + 5 等于多少？"}]}
            )
            print(f"回答: {response['messages'][-1].content}\n")

    except ImportError:
        print("请先安装依赖: pip install langchain-mcp-adapters mcp")
    except Exception as e:
        print(f"运行失败（需要先启动 MCP 服务器）: {e}\n")


# ============================================================
# 主函数
# ============================================================
async def main():
    print("Demo 12: 模型上下文协议 (MCP)\n")
    print("MCP 是一种开放协议，标准化了应用程序如何向 LLM 提供工具和上下文。")
    print("LangChain 代理可以使用 langchain-mcp-adapters 库使用 MCP 服务器上的工具。\n")

    print("=" * 60)
    print("第一部分：自定义 MCP 服务器代码")
    print("=" * 60)
    print("\n--- math_server.py (stdio 传输) ---")
    print(MATH_SERVER_CODE)
    print("\n--- weather_server.py (Streamable HTTP 传输) ---")
    print(WEATHER_SERVER_CODE)

    print("\n使用方法:")
    print("1. 安装依赖: pip install langchain-mcp-adapters mcp")
    print("2. 在一个终端启动数学服务器: python math_server.py")
    print("3. 在另一个终端启动天气服务器: python weather_server.py")
    print("4. 运行本 demo 的 MCP 工具演示\n")

    # 演示 MCP 工具
    await demo_mcp_tools()

    # 演示有状态 MCP
    await demo_stateful_mcp()

    print("\nMCP 总结:")
    print("1. 传输类型: stdio(本地)、streamable_http(远程)、SSE(实时流)")
    print("2. MultiServerMCPClient 默认无状态: 每次工具调用创建新会话")
    print("3. 有状态模式: 使用 client.session() 创建持久会话")
    print("4. 自定义服务器: 使用 FastMCP 创建工具并运行")
    print("5. MCP 标准化了 AI 应用如何获取工具和上下文")


if __name__ == "__main__":
    asyncio.run(main())
