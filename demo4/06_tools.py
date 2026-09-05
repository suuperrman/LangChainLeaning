"""
LangChain V1 入门示例 06 —— 工具 (Tools)

对应文档：
  - https://langchain-doc.cn/v1/python/langchain/tools.html

核心知识点：
  1. @tool 装饰器：最简单的工具定义方式
  2. 自定义工具属性：名称、描述
  3. 高级架构定义：Pydantic / JSON Schema 定义复杂输入
  4. ToolRuntime：工具访问运行时上下文（state、context、store、stream_writer）
  5. Command：从工具返回状态更新
  6. 工具与智能体结合使用

注意：
  - 工具的类型提示（type hints）是必需的，它们定义了工具的输入架构
  - 文档字符串（docstring）会成为工具的描述，帮助模型理解何时使用
  - ToolRuntime 参数对模型是隐藏的，不会出现在工具 schema 中
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Any

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

# 统一的模型创建方式（与 demo2 保持一致，使用阿里云通义千问）
def create_model(temperature: float = 0.4) -> ChatOpenAI:
    """创建 ChatOpenAI 模型实例（使用阿里云 DashScope 兼容接口）。"""
    return ChatOpenAI(
        model="qwen3.8-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        # 关闭 qwen 的思考模式，否则会报错
        extra_body={"enable_thinking": False},
    )

# ============================================================
# 一、基础工具定义
# ============================================================
print("=" * 60)
print("一、基础工具定义（@tool 装饰器）")
print("=" * 60)


@tool
def search_database(query: str, limit: int = 10) -> str:
    """搜索客户数据库，返回匹配的记录。

    Args:
        query: 搜索关键词
        limit: 返回结果的最大数量（默认10条）
    """
    # 模拟数据库查询
    mock_results = [f"用户{i}: 姓名{query}_{i}" for i in range(limit)]
    return f"找到 {limit} 条关于 '{query}' 的记录:\n" + "\n".join(mock_results)


print(f"工具名称: {search_database.name}")
print(f"工具描述: {search_database.description[:50]}...")
print(f"工具输入参数: {list(search_database.args.keys())}")

# 工具可以直接调用（不通过模型）
result = search_database.invoke({"query": "张三", "limit": 3})
print(f"\n直接调用工具结果:\n{result}")
print()

# ============================================================
# 二、自定义工具属性
# ============================================================
print("=" * 60)
print("二、自定义工具属性")
print("=" * 60)


@tool("weather_checker", description="查询指定城市的天气信息。当用户问天气时使用此工具。")
def get_weather(city: str) -> str:
    """获取天气信息。"""
    weather_data = {
        "北京": "晴朗，25°C",
        "上海": "多云，22°C",
        "广州": "小雨，28°C",
        "深圳": "晴转多云，27°C",
    }
    return weather_data.get(city, f"{city} 的天气：暂无数据")


print(f"工具名称: {get_weather.name}")
print(f"工具描述: {get_weather.description}")
print(f"输入参数: {get_weather.args}")
print()

# ============================================================
# 三、高级架构定义（Pydantic Schema）
# ============================================================
print("=" * 60)
print("三、高级架构定义（Pydantic Schema）")
print("=" * 60)


class WeatherInput(BaseModel):
    """天气查询的输入参数。"""
    location: str = Field(description="城市名称或坐标，如 '北京' 或 '31.2N,121.5E'")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位，摄氏度或华氏度",
    )
    include_forecast: bool = Field(
        default=False,
        description="是否包含未来5天的天气预报",
    )


@tool(args_schema=WeatherInput)
def get_detailed_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """获取当前天气及可选的天气预报。"""
    temp_c = 22
    temp = temp_c if units == "celsius" else temp_c * 9 / 5 + 32
    unit_label = "°C" if units == "celsius" else "°F"

    result = f"{location} 当前天气：{temp}{unit_label}，晴朗"
    if include_forecast:
        result += "\n未来5天预报：周一天晴，周二多云，周三小雨，周四晴，周五多云"
    return result


print(f"工具名称: {get_detailed_weather.name}")
print("Pydantic Schema 定义的参数:")
for field_name, field_info in WeatherInput.model_fields.items():
    print(f"  - {field_name}: {field_info.annotation} (默认: {field_info.default})")

# 测试调用
result = get_detailed_weather.invoke({
    "location": "上海",
    "units": "fahrenheit",
    "include_forecast": True,
})
print(f"\n调用结果:\n{result}")
print()

# ============================================================
# 四、工具与智能体结合
# ============================================================
print("=" * 60)
print("四、工具与智能体结合（ReAct 模式）")
print("=" * 60)


@tool
def calculate(expression: str) -> str:
    """执行数学计算。支持加减乘除，如 '2 + 3 * 4'。仅用于数学计算。"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# 创建带工具的智能体
agent = create_agent(
    model=create_model(temperature=0.3),
    tools=[get_weather, calculate],
    system_prompt="你是一个生活助手，可以查询天气和进行计算。",
)

print("智能体已创建，包含工具: get_weather, calculate")
print("智能体会自动决定何时调用哪个工具（ReAct 模式）")
print()

# 测试：一个需要调用天气工具的问题
print("--- 测试1：天气查询（应自动调用 get_weather 工具）---")
response = agent.invoke({"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]})
print(f"最终回答: {response['messages'][-1].content}")
print()

# 测试：一个需要调用计算工具的问题
print("--- 测试2：数学计算（应自动调用 calculate 工具）---")
response = agent.invoke({"messages": [{"role": "user", "content":" 128 * 256 等于多少？"}]})
print(f"最终回答: {response['messages'][-1].content}")
print()

# 测试：不需要工具的问题
print("--- 测试3：纯问答（不需要调用工具）---")
response = agent.invoke({"messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}]})
print(f"最终回答: {response['messages'][-1].content}")
print()

# ============================================================
# 五、ToolRuntime —— 访问运行时上下文
# ============================================================
print("=" * 60)
print("五、ToolRuntime —— 工具访问运行时上下文")
print("=" * 60)

print("ToolRuntime 提供以下能力：")
print("  - runtime.state       : 访问代理状态（消息、自定义字段）")
print("  - runtime.context     : 访问不可变上下文（用户ID、配置等）")
print("  - runtime.store       : 访问跨对话持久存储")
print("  - runtime.stream_writer: 在工具执行时流式传输自定义更新")
print("  - runtime.tool_call_id: 当前工具调用的 ID")
print()
print("注意：ToolRuntime 参数对模型是隐藏的，不会出现在工具 schema 中")
print()


@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """总结当前对话的统计信息。"""
    messages = runtime.state["messages"]

    human_count = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_count = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_count = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return (
        f"对话统计：共 {len(messages)} 条消息，"
        f"其中用户消息 {human_count} 条，"
        f"AI 回复 {ai_count} 条，"
        f"工具结果 {tool_count} 条。"
    )


print("工具 summarize_conversation 可以访问 runtime.state 中的消息历史")
print("（该工具需要在智能体上下文中运行，见后续示例）")
print()

# ============================================================
# 六、使用 Command 从工具更新状态
# ============================================================
print("=" * 60)
print("六、Command —— 从工具返回状态更新")
print("=" * 60)

print("使用 langgraph.types.Command 可以在工具中更新代理状态")
print("常见用途：")
print("  - 更新用户偏好设置")
print("  - 清除对话历史")
print("  - 控制图的执行流程")
print()

# 导入 Command（需要 langgraph 包）
try:
    from langgraph.types import Command
    from langchain.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    @tool
    def clear_history(runtime: ToolRuntime) -> Command:
        """清除当前的对话历史记录。"""
        return Command(
            update={
                "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            }
        )

    @tool
    def set_user_preference(key: str, value: str, runtime: ToolRuntime) -> Command:
        """设置用户的偏好配置。

        Args:
            key: 偏好项的键名，如 'theme'、'language'
            value: 偏好项的值
        """
        # 获取当前的 preferences 字典（如果不存在则为空字典）
        current_prefs = runtime.state.get("user_preferences", {})
        current_prefs[key] = value

        return Command(
            update={"user_preferences": current_prefs}
        )

    print("已创建 clear_history 和 set_user_preference 工具")
    print("（这些工具需要在智能体上下文中运行，见短期记忆示例）")
except ImportError:
    print("需要安装 langgraph 才能使用 Command，请先运行: pip install langgraph")

print()
print("提示：ToolRuntime 的完整用法将在后续的 '短期记忆' 和 '流式传输' 示例中演示")
