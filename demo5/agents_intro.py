"""
LangChain V1 入门示例 07 —— 智能体 (Agents) 基础

对应文档：
  - https://langchain-doc.cn/v1/python/langchain/agents.html

核心知识点：
  1. create_agent：创建生产就绪的智能体
  2. 核心组件：模型 + 工具 + 系统提示
  3. 静态模型 vs 动态模型选择
  4. ReAct 循环：推理(Thought) -> 行动(Action) -> 观察(Observation)
  5. 系统提示：字符串方式 / 动态方式
  6. 智能体的调用方式：invoke
  7. 智能体状态：messages 等字段

核心概念：
  智能体 = LLM + 工具 + 循环推理
  - 模型（推理引擎）：决定做什么
  - 工具（行动能力）：执行具体操作
  - ReAct 循环：推理->行动->观察，直到得出答案
  - 系统提示：塑造智能体的行为方式
"""

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, ToolMessage

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
# 一、准备工具
# ============================================================
# 先定义几个模拟工具，用于演示智能体的工具调用能力

@tool
def search_product(query: str) -> str:
    """搜索产品数据库，返回匹配的产品列表。

    Args:
        query: 产品搜索关键词
    """
    products = {
        "耳机": [
            {"name": "WH-1000XM5", "price": 2999, "brand": "Sony"},
            {"name": "AirPods Pro 2", "price": 1899, "brand": "Apple"},
            {"name": "Bose QC Ultra", "price": 3299, "brand": "Bose"},
        ],
        "手机": [
            {"name": "iPhone 15 Pro", "price": 7999, "brand": "Apple"},
            {"name": "小米14", "price": 3999, "brand": "Xiaomi"},
        ],
    }
    for key, items in products.items():
        if key in query:
            return f"找到 {len(items)} 款{key}：\n" + "\n".join(
                f"- {p['name']} ({p['brand']}) - ¥{p['price']}" for p in items
            )
    return f"未找到 '{query}' 相关产品"


@tool
def check_inventory(product_name: str) -> str:
    """查询指定产品的库存状态。

    Args:
        product_name: 产品名称（精确名称）
    """
    inventory = {
        "WH-1000XM5": 15,
        "AirPods Pro 2": 0,
        "Bose QC Ultra": 8,
        "iPhone 15 Pro": 23,
        "小米14": 50,
    }
    stock = inventory.get(product_name)
    if stock is None:
        return f"产品 '{product_name}' 不存在"
    elif stock > 0:
        return f"{product_name} 当前库存：{stock} 件"
    else:
        return f"{product_name} 当前缺货"


@tool
def calculate_price(base_price: float, discount: float = 0.0) -> str:
    """计算折扣后的最终价格。

    Args:
        base_price: 原价
        discount: 折扣率（0.1 表示 9折，0.2 表示 8折，默认 0 表示无折扣）
    """
    final_price = base_price * (1 - discount)
    return f"原价 ¥{base_price}，折扣 {discount*10:.0f} 折，最终价格 ¥{final_price:.2f}"


print("已准备 3 个工具：search_product、check_inventory、calculate_price")
print()

# ============================================================
# 二、创建最基础的智能体
# ============================================================
print("=" * 60)
print("二、创建基础智能体（静态模型 + 工具）")
print("=" * 60)

# 方式 1：用模型实例创建（与 demo2 保持一致）
agent = create_agent(
    model=create_model(temperature=0.3),  # 模型（静态，创建时固定）
    tools=[search_product, check_inventory, calculate_price],  # 工具列表
    system_prompt="你是一个专业的电商客服助手，帮助用户查询产品和库存。回答要简洁准确。",
)

print("智能体创建成功！")
print(f"  模型: qwen3.8-max (阿里云通义千问)")
print(f"  工具数量: 3")
print(f"  系统提示: 电商客服助手")
print()

# ============================================================
# 三、调用智能体 & 理解 ReAct 循环
# ============================================================
print("=" * 60)
print("三、调用智能体 —— 理解 ReAct 循环")
print("=" * 60)

print("ReAct = Reasoning (推理) + Acting (行动)")
print("智能体循环过程：")
print("  1. 模型推理：思考需要做什么")
print("  2. 选择工具：决定调用哪个工具及参数")
print("  3. 执行工具：工具返回结果(Observation)")
print("  4. 再次推理：根据结果决定下一步")
print("  5. 重复...直到得出最终答案")
print()

# 测试一个需要多步工具调用的问题
question = "帮我查一下无线耳机有哪些款式，然后看看 Sony WH-1000XM5 有没有货，最后算一下打8折后的价格"
print(f"用户问题: {question}")
print()

result = agent.invoke({"messages": [{"role": "user", "content": question}]})

# 查看完整的消息历史（展示 ReAct 循环过程）
print("--- ReAct 循环过程 ---")
for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    if msg_type == "HumanMessage":
        print(f"\n[{i}] 👤 用户: {msg.content[:50]}...")
    elif msg_type == "AIMessage":
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n[{i}] 🤖 AI 决定调用工具: {tc['name']}({tc['args']})")
        else:
            print(f"\n[{i}] 🤖 AI 最终回答: {msg.content[:80]}...")
    elif msg_type == "ToolMessage":
        print(f"[{i}] 🔧 工具结果 ({msg.name}): {msg.content[:60]}...")
print()

print("--- 最终回答 ---")
print(result["messages"][-1].content)
print()

# ============================================================
# 四、智能体状态结构
# ============================================================
print("=" * 60)
print("四、智能体状态 (AgentState)")
print("=" * 60)

print("智能体的状态是一个字典，包含以下关键字段：")
print(f"  - messages: 消息列表（共 {len(result['messages'])} 条）")
print(f"  - 其他字段可通过 state_schema 自定义扩展")
print()

# 查看状态的键
print("状态字典的键:", list(result.keys()))
print()

# ============================================================
# 五、动态系统提示
# ============================================================
print("=" * 60)
print("五、动态系统提示")
print("=" * 60)

print("系统提示也可以是动态的，根据运行时上下文生成。")
print("使用 @dynamic_prompt 装饰器实现（详见中间件示例）")
print()

# ============================================================
# 六、动态模型选择（进阶）
# ============================================================
print("=" * 60)
print("六、动态模型选择（进阶）")
print("=" * 60)

print("动态模型根据当前状态和上下文在运行时选择模型。")
print("使用 @wrap_model_call 中间件装饰器实现：")
print("""
  from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

  @wrap_model_call
  def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
      message_count = len(request.state["messages"])
      if message_count > 10:
          request.model = create_model(temperature=0.7)  # 长对话用高创造力
      else:
          request.model = create_model(temperature=0.3)  # 短对话用精准模式
      return handler(request)

  agent = create_agent(
      model=create_model(),
      tools=tools,
      middleware=[dynamic_model_selection]
  )
""")
print("（完整示例见中间件章节）")
print()

# ============================================================
# 七、空工具列表 = 纯 LLM 节点
# ============================================================
print("=" * 60)
print("七、空工具列表 = 纯 LLM 节点")
print("=" * 60)

# 如果工具列表为空，智能体就只是一个带系统提示的 LLM 包装器
simple_agent = create_agent(
    model=create_model(temperature=0.5),
    tools=[],  # 空工具列表
    system_prompt="你是一个简短回答助手，用一句话回答问题。",
)

response = simple_agent.invoke({"messages": [{"role": "user", "content": "什么是微服务？"}]})
print("空工具智能体的回答:")
print(f"  {response['messages'][-1].content}")
print()
print("提示：即使没有工具，create_agent 仍然提供以下好处：")
print("  - 统一的状态管理")
print("  - 中间件支持")
print("  - 流式传输支持")
print("  - 便于后续添加工具扩展能力")
