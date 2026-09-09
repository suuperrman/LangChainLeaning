"""
LangChain V1 入门示例 08 —— 结构化输出 (Structured Output)

对应文档：
  - https://langchain-doc.cn/v1/python/langchain/structured-output.html

核心知识点：
  1. 什么是结构化输出：让智能体返回可预测格式的数据
  2. 两种策略：ProviderStrategy（提供商原生）vs ToolStrategy（工具调用）
  3. 四种 schema 定义方式：Pydantic / Dataclass / TypedDict / JSON Schema
  4. response_format 参数配置
  5. structured_response 返回值
  6. 错误处理：handle_errors 参数
  7. 自定义工具消息内容：tool_message_content

使用场景：
  - 信息抽取（从文本中提取结构化数据）
  - 分类任务（返回固定类别的结果）
  - API 接口返回（直接给前端/后端使用）
  - 数据验证（确保输出符合预期格式）
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Literal, List, Optional
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
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
# 一、Pydantic 模式（最推荐，功能最完整）
# ============================================================
print("=" * 60)
print("一、Pydantic 模式（推荐）")
print("=" * 60)


class ContactInfo(BaseModel):
    """一个人的联系信息。"""
    name: str = Field(description="该人的姓名")
    email: str = Field(description="该人的电子邮件地址")
    phone: str = Field(description="该人的电话号码")


# 直接传入 Pydantic 类，LangChain 自动选择最佳策略
# OpenAI 模型会用 ProviderStrategy（原生结构化输出，最可靠）
agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=ContactInfo,  # 自动选择策略
)

text = """
张三，男，30岁，软件工程师。
邮箱：zhangsan@example.com
电话：138-0000-1234
地址：北京市朝阳区
"""

result = agent.invoke({
    "messages": [{"role": "user", "content": f"从以下文本中提取联系信息：\n{text}"}]
})

# 结构化结果在 structured_response 字段中
print("提取结果 (Pydantic 对象):")
print(f"  name : {result['structured_response'].name}")
print(f"  email: {result['structured_response'].email}")
print(f"  phone: {result['structured_response'].phone}")
print(f"  类型 : {type(result['structured_response']).__name__}")
print()

# 可以直接转为字典
print("转为字典:")
print(result["structured_response"].model_dump())
print()

# ============================================================
# 二、Dataclass 模式
# ============================================================
print("=" * 60)
print("二、Dataclass 模式")
print("=" * 60)


@dataclass
class BookInfo:
    """一本书的基本信息。"""
    title: str           # 书名
    author: str          # 作者
    year: int            # 出版年份
    genre: str           # 类型/流派


agent_dataclass = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=BookInfo,
)

result = agent_dataclass.invoke({
    "messages": [{"role": "user", "content": "提取《三体》的信息：作者刘慈欣，2008年出版，科幻小说"}]
})

book = result["structured_response"]
print(f"提取结果 (Dataclass):")
print(f"  title : {book.title}")
print(f"  author: {book.author}")
print(f"  year  : {book.year}")
print(f"  genre : {book.genre}")
print()

# ============================================================
# 三、TypedDict 模式
# ============================================================
print("=" * 60)
print("三、TypedDict 模式")
print("=" * 60)


class MovieInfo(TypedDict):
    """电影信息。"""
    title: str       # 电影名称
    director: str    # 导演
    rating: float    # 评分（0-10）
    duration: int    # 时长（分钟）


agent_typeddict = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=MovieInfo,
)

result = agent_typeddict.invoke({
    "messages": [{"role": "user", "content": "《盗梦空间》由克里斯托弗·诺兰执导，IMDb评分8.8，时长148分钟"}]
})

# TypedDict 返回的是普通字典
movie = result["structured_response"]
print(f"提取结果 (TypedDict = 普通字典):")
print(f"  title   : {movie['title']}")
print(f"  director: {movie['director']}")
print(f"  rating  : {movie['rating']}")
print(f"  duration: {movie['duration']}")
print(f"  类型    : {type(movie).__name__}")
print()

# ============================================================
# 四、JSON Schema 模式
# ============================================================
print("=" * 60)
print("四、JSON Schema 模式")
print("=" * 60)

# 直接用 JSON Schema 字典定义
product_schema = {
    "type": "object",
    "description": "产品信息。",
    "properties": {
        "name": {"type": "string", "description": "产品名称"},
        "price": {"type": "number", "description": "产品价格（元）"},
        "category": {"type": "string", "description": "产品类别"},
        "in_stock": {"type": "boolean", "description": "是否有库存"},
    },
    "required": ["name", "price", "category", "in_stock"],
}

agent_json_schema = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=product_schema,
)

result = agent_json_schema.invoke({
    "messages": [{"role": "user", "content": "描述这款产品：iPhone 15 Pro，售价7999元，手机类，目前有货"}]
})

product = result["structured_response"]
print(f"提取结果 (JSON Schema = 字典):")
print(f"  name    : {product['name']}")
print(f"  price   : {product['price']}")
print(f"  category: {product['category']}")
print(f"  in_stock: {product['in_stock']}")
print()

# ============================================================
# 五、复杂嵌套结构 + 列表
# ============================================================
print("=" * 60)
print("五、复杂嵌套结构 + 列表")
print("=" * 60)


class OrderItem(BaseModel):
    """订单中的商品项。"""
    product_name: str = Field(description="商品名称")
    quantity: int = Field(description="数量", ge=1)
    unit_price: float = Field(description="单价（元）", gt=0)


class OrderSummary(BaseModel):
    """订单摘要信息。"""
    order_id: str = Field(description="订单编号")
    customer_name: str = Field(description="客户姓名")
    items: List[OrderItem] = Field(description="商品列表")
    total_amount: float = Field(description="订单总金额", ge=0)
    status: Literal["pending", "paid", "shipped", "delivered"] = Field(
        description="订单状态"
    )
    shipping_address: Optional[str] = Field(default=None, description="收货地址")


agent_order = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=OrderSummary,
)

order_text = """
订单号 ORD202401001，客户李四。
商品：
  - MacBook Pro 1台，单价14999元
  - Magic Mouse 2个，单价699元
  - USB-C 转接线 3条，单价129元
总金额 16784元，已付款，等待发货。
收货地址：上海市浦东新区张江高科技园区。
"""

result = agent_order.invoke({
    "messages": [{"role": "user", "content": f"从以下订单文本中提取结构化信息：\n{order_text}"}]
})

order = result["structured_response"]
print(f"订单编号: {order.order_id}")
print(f"客户姓名: {order.customer_name}")
print(f"订单状态: {order.status}")
print(f"商品列表 ({len(order.items)} 项):")
for item in order.items:
    subtotal = item.quantity * item.unit_price
    print(f"  - {item.product_name} x {item.quantity} = ¥{subtotal:.2f}")
print(f"总金额: ¥{order.total_amount:.2f}")
print(f"收货地址: {order.shipping_address}")
print()

# ============================================================
# 六、ToolStrategy 显式指定 + 错误处理
# ============================================================
print("=" * 60)
print("六、ToolStrategy 显式指定 + 错误处理")
print("=" * 60)

print("ToolStrategy 使用工具调用来实现结构化输出")
print("适用于不支持原生结构化输出的模型")
print()

print("错误处理策略 handle_errors:")
print("  - True           : 默认，捕获所有错误并重试")
print("  - False          : 不处理，异常直接抛出")
print("  - str            : 自定义错误消息")
print("  - Exception 类型  : 仅处理指定异常类型")
print("  - callable       : 自定义错误处理函数")
print()


class ProductRating(BaseModel):
    """产品评分。"""
    rating: int = Field(description="评分，1-5分", ge=1, le=5)
    comment: str = Field(description="评论内容")


# 显式使用 ToolStrategy，并自定义错误消息
agent_rating = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors="请提供 1-5 之间的有效整数评分，并包含评论内容。",
    ),
)

result = agent_rating.invoke({
    "messages": [{"role": "user", "content": "评价这款耳机：音质很好，降噪出色，性价比高，给4.5分"}]
})

rating = result["structured_response"]
print(f"评分: {rating.rating} / 5")
print(f"评论: {rating.comment}")
print()

# ============================================================
# 七、自定义工具消息内容
# ============================================================
print("=" * 60)
print("七、自定义工具消息内容")
print("=" * 60)

print("tool_message_content 参数可自定义生成结构化输出时")
print("对话历史中显示的工具消息内容")
print()


class ActionItem(BaseModel):
    """从会议记录中提取的行动事项。"""
    task: str = Field(description="需要完成的具体任务")
    assignee: str = Field(description="负责人")
    priority: Literal["low", "medium", "high"] = Field(description="优先级")


agent_action = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=ToolStrategy(
        schema=ActionItem,
        tool_message_content="行动事项已成功捕获并添加到待办列表！",
    ),
)

result = agent_action.invoke({
    "messages": [{"role": "user", "content": "会议决定：小王需要在本周五前完成项目进度汇报PPT"}]
})

action = result["structured_response"]
print(f"任务: {action.task}")
print(f"负责人: {action.assignee}")
print(f"优先级: {action.priority}")
print()
print("提示：如果不设置 tool_message_content，工具消息会显示结构化数据本身")
print("设置后则显示自定义内容，对话历史更自然")
print()

# ============================================================
# 八、结构化输出 + 工具结合使用
# ============================================================
print("=" * 60)
print("八、结构化输出 + 工具结合使用")
print("=" * 60)

print("智能体可以先调用工具获取数据，再以结构化格式返回结果")
print("这是实际项目中最常见的用法")
print()

from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询城市的天气信息。"""
    data = {
        "北京": "25°C，晴朗，湿度45%，风速3级",
        "上海": "22°C，多云，湿度60%，风速2级",
        "广州": "28°C，小雨，湿度80%，风速4级",
    }
    return data.get(city, f"{city}：暂无数据")


class WeatherData(BaseModel):
    """结构化的天气数据。"""
    city: str = Field(description="城市名称")
    temperature: float = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况，如晴朗、多云、小雨等")
    humidity: int = Field(description="湿度百分比", ge=0, le=100)


agent_weather = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    response_format=WeatherData,
    system_prompt="你是一个天气查询助手，先调用工具获取数据，再以结构化格式返回。",
)

result = agent_weather.invoke({
    "messages": [{"role": "user", "content": "查询北京的天气"}]
})

weather = result["structured_response"]
print(f"城市: {weather.city}")
print(f"温度: {weather.temperature}°C")
print(f"天气: {weather.condition}")
print(f"湿度: {weather.humidity}%")
print()
print("注意：智能体先调用了 get_weather 工具获取数据，")
print("然后将数据解析为 WeatherData 结构化对象返回")
