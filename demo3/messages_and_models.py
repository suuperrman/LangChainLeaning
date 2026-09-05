"""
LangChain V1 入门示例 05 —— 消息 (Messages) 与模型 (Models)

对应文档：
  - https://langchain-doc.cn/v1/python/langchain/messages.html
  - https://langchain-doc.cn/v1/python/langchain/models.html

核心知识点：
  1. 四种消息类型：SystemMessage / HumanMessage / AIMessage / ToolMessage
  2. 三种消息输入格式：文本提示 / 消息对象 / 字典格式
  3. 模型初始化：init_chat_model vs ChatOpenAI 直接构造
  4. 三种调用方式：invoke / stream / batch
  5. AIMessage 的属性：content、tool_calls、usage_metadata
  6. 消息元数据：name、id

注意：本文件使用 LangChain V1 API（langchain.messages 模块）
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

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
# 一、模型初始化
# ============================================================
print("=" * 60)
print("一、模型初始化")
print("=" * 60)

# 底层通过阿里云 DashScope 的 OpenAI 兼容接口调用通义千问 qwen3.8-max
model = create_model(temperature=0.3)
print(f"模型: {model.model_name}")
print(f"提供商: 阿里云 DashScope (通义千问)")
print(f"接口: OpenAI 兼容模式")

# 常用参数说明：
#   temperature   : 0=确定性输出, 越高越有创造性
#   max_tokens    : 最大输出 token 数
#   timeout       : 请求超时时间(秒)
#   base_url      : API 地址（DashScope 兼容模式）
#   extra_body    : 额外参数（如关闭思考模式）
print()

# ============================================================
# 二、四种消息类型
# ============================================================
print("=" * 60)
print("二、四种消息类型")
print("=" * 60)

# --- 1. SystemMessage：系统消息，设定角色/行为 ---
sys_msg = SystemMessage(
    "你是一个资深 Python 开发工程师，回答要简洁准确，优先给出代码示例。"
)
print(f"1. SystemMessage - 设定模型角色: {sys_msg.content[:30]}...")

# --- 2. HumanMessage：人类消息，表示用户输入 ---
human_msg = HumanMessage(content="什么是列表推导式？给个例子。")
print(f"2. HumanMessage - 用户输入: {human_msg.content[:20]}...")

# HumanMessage 可以带元数据（name 标识不同用户，id 用于追踪）
human_msg_with_meta = HumanMessage(
    content="你好！",
    name="alice",   # 标识发送者
    id="msg_001",   # 唯一标识符
)
print(f"   带元数据的 HumanMessage: name={human_msg_with_meta.name}, id={human_msg_with_meta.id}")

# --- 3. AIMessage：AI 消息，表示模型回复 ---
# 通常由模型返回，但也可以手动构造（如模拟历史对话）
ai_msg_manual = AIMessage("好的，我来为你解释列表推导式。")
print(f"3. AIMessage - 模型回复(手动构造): {ai_msg_manual.content[:20]}...")

# --- 4. ToolMessage：工具消息，表示工具调用的返回结果 ---
# 用于工具调用场景，需要 tool_call_id 与对应 AI 消息中的工具调用关联
tool_msg = ToolMessage(
    content="北京当前温度：25°C，天气晴朗",
    tool_call_id="call_abc123",
    name="get_weather",
)
print(f"4. ToolMessage - 工具结果: {tool_msg.content[:20]}...")
print(f"   tool_call_id={tool_msg.tool_call_id}, name={tool_msg.name}")
print()

# ============================================================
# 三、三种消息输入格式
# ============================================================
print("=" * 60)
print("三、三种消息输入格式")
print("=" * 60)

# --- 格式 1：纯文本（最简单，单轮对话） ---
print("【格式1】纯文本提示（内部自动转为 HumanMessage）")
response = model.invoke("用一句话解释什么是闭包。")
print(f"  回答: {response.content}")
print()

# --- 格式 2：消息对象列表（最灵活，多轮对话必备） ---
print("【格式2】消息对象列表（支持多轮对话和系统提示）")
messages = [
    SystemMessage("你是一个简短回答助手，每句话不超过 20 个字。"),
    HumanMessage("什么是装饰器？"),
]
response = model.invoke(messages)
print(f"  回答: {response.content}")
print()

# --- 格式 3：字典格式（OpenAI 原生格式，方便从 JSON 加载） ---
print("【格式3】字典格式（与 OpenAI API 格式一致）")
messages_dict = [
    {"role": "system", "content": "你是一个简短回答助手。"},
    {"role": "user", "content": "什么是生成器？"},
]
response = model.invoke(messages_dict)
print(f"  回答: {response.content}")
print()

# ============================================================
# 四、三种调用方式
# ============================================================
print("=" * 60)
print("四、三种调用方式")
print("=" * 60)

# --- 1. invoke：单次同步调用，返回完整结果 ---
print("【1】invoke - 单次同步调用")
response = model.invoke("解释什么是 GIL。")
print(f"  回答: {response.content[:80]}...")
print()

# --- AIMessage 的重要属性 ---
print("  AIMessage 属性一览:")
print(f"    type          : {response.type}")     # 'ai'
print(f"    content       : 文本内容")
print(f"    tool_calls    : {response.tool_calls}")  # 工具调用列表（本示例为空）
# usage_metadata 需要模型返回 token 使用信息
if response.usage_metadata:
    print(f"    usage_metadata: 输入={response.usage_metadata['input_tokens']}, "
          f"输出={response.usage_metadata['output_tokens']}, "
          f"总计={response.usage_metadata['total_tokens']}")
print()

# --- 2. stream：流式输出，逐 token 返回 ---
print("【2】stream - 流式输出（实时显示）")
print("  流式输出: ", end="", flush=True)
for chunk in model.stream("用三句话介绍 Python 的特点。"):
    print(chunk.content, end="", flush=True)
print("\n")

# --- 3. batch：批量调用，并行处理多个输入 ---
print("【3】batch - 批量调用（并行处理）")
batch_inputs = [
    [HumanMessage("Python 和 Java 的最大区别？")],
    [HumanMessage("什么是虚拟环境？")],
    [HumanMessage("解释一下鸭子类型。")],
]
results = model.batch(batch_inputs, config={"max_concurrency": 3})
for i, res in enumerate(results, 1):
    print(f"  问题 {i} 回答: {res.content[:50]}...")
print()

# ============================================================
# 五、手动构造多轮对话历史
# ============================================================
print("=" * 60)
print("五、手动构造多轮对话历史")
print("=" * 60)

# 有时你需要手动插入 AI 消息来模拟历史对话
# 例如从数据库加载历史记录后重建对话
conversation = [
    SystemMessage("你是一个友善的编程导师。"),
    HumanMessage("我想学 Python，应该从哪开始？"),
    AIMessage("建议从基础语法开始：变量、数据类型、条件语句、循环。"),
    HumanMessage("好的，那循环有几种？"),
]

response = model.invoke(conversation)
print(f"用户: 好的，那循环有几种？")
print(f"助手: {response.content}")
