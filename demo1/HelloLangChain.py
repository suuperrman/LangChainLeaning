from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="qwen3.8-max",base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",temperature=0.4)

def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"


agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手",
)

# 运行代理
res = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]}
)

print(res)