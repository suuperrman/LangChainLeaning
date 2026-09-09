"""
Demo 15: 检索 (Retrieval / RAG)
=================================
检索通过在查询时获取相关的外部知识，解决 LLM 上下文有限和知识静态的问题。

核心知识点：
1. 检索管道: 文档加载 -> 切分 -> 嵌入 -> 向量存储 -> 检索器
2. 三种 RAG 架构:
   - 2 步 RAG: 检索总是在生成之前执行（简单、可预测）
   - Agentic RAG: 智能体决定何时和如何检索（灵活）
   - 混合 RAG: 结合两者特点，带验证步骤
3. 构成要素: 文档加载器、文本切分器、嵌入模型、向量存储、检索器

参考文档: https://langchain-doc.cn/v1/python/langchain/retrieval.html
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


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
# 示例知识库数据
# ============================================================
SAMPLE_DOCUMENTS = [
    {
        "content": "Python 是一种高级编程语言，以简洁的语法和强大的标准库著称。"
                   "它支持多种编程范式，包括面向对象、函数式和过程式编程。"
                   "Python 广泛应用于 Web 开发、数据分析、人工智能和自动化脚本。",
        "source": "python_intro.txt"
    },
    {
        "content": "LangChain 是一个用于构建 LLM 应用的框架。"
                   "它提供了模块化的组件，包括文档加载器、文本切分器、嵌入模型、向量存储和检索器。"
                   "LangChain 的核心抽象是智能体（Agent），可以调用工具来完成复杂任务。",
        "source": "langchain_intro.txt"
    },
    {
        "content": "向量数据库是专门用于存储和搜索嵌入向量的数据库。"
                   "常见的向量数据库包括 FAISS、Chroma、Pinecone 和 Weaviate。"
                   "它们通过余弦相似度等度量方式找到含义相似的文本。",
        "source": "vector_db.txt"
    },
    {
        "content": "RAG（检索增强生成）通过在查询时获取相关的外部知识来增强 LLM 的回答。"
                   "RAG 解决了 LLM 上下文有限和知识静态的问题。"
                   "RAG 有多种实现方式：2 步 RAG、Agentic RAG 和混合 RAG。",
        "source": "rag_intro.txt"
    },
    {
        "content": "智能体（Agent）是 LLM 驱动的自主系统，可以调用工具进行推理和决策。"
                   "LangChain 的 create_agent 创建的智能体支持工具调用、结构化输出、"
                   "流式传输、短期记忆和长期记忆等功能。",
        "source": "agent_intro.txt"
    }
]


# ============================================================
# 第一部分：构建简易知识库
# ============================================================
# 模拟文档加载和切分的过程

def build_knowledge_base():
    """构建简易知识库。"""
    print("=" * 60)
    print("第一部分：构建知识库")
    print("=" * 60)

    print("\n检索管道流程:")
    print("  来源(文件/API/数据库)")
    print("    -> 文档加载器(Document Loaders)")
    print("    -> 文档(Documents)")
    print("    -> 切分成块(Split into chunks)")
    print("    -> 转化为嵌入(Turn into embeddings)")
    print("    -> 向量存储(Vector Store)")
    print("    -> 检索器(Retriever)")
    print("    -> LLM 使用检索到的信息")

    print(f"\n知识库包含 {len(SAMPLE_DOCUMENTS)} 篇文档:")
    for doc in SAMPLE_DOCUMENTS:
        print(f"  - {doc['source']}: {doc['content'][:30]}...")

    print()
    return SAMPLE_DOCUMENTS


# ============================================================
# 第二部分：简易检索器（关键词匹配模拟向量检索）
# ============================================================

def simple_retriever(query: str, documents: list) -> list:
    """简易检索器：基于关键词匹配模拟向量检索。"""
    results = []
    query_words = set(query.lower().replace("？", "").replace("?", "").split())

    for doc in documents:
        content_lower = doc["content"].lower()
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            results.append((doc, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:3]]


# ============================================================
# 第三部分：2 步 RAG
# ============================================================
# 检索步骤总是在生成步骤之前执行
# 简单且可预测，适用于 FAQ、文档机器人

def demo_two_step_rag():
    """演示 2 步 RAG。"""
    print("=" * 60)
    print("第三部分：2 步 RAG（检索 -> 生成）")
    print("=" * 60)

    query = "什么是 Python？"

    # 步骤1: 检索相关文档
    print(f"\n步骤1: 检索（查询: '{query}'）")
    retrieved_docs = simple_retriever(query, SAMPLE_DOCUMENTS)
    context = "\n\n".join([f"[{doc['source']}]: {doc['content']}" for doc in retrieved_docs])
    print(f"检索到 {len(retrieved_docs)} 篇相关文档")

    # 步骤2: 生成答案
    print("\n步骤2: 生成答案")
    model = create_model(temperature=0.3)
    prompt = f"""请根据以下检索到的信息回答问题。

检索到的信息:
{context}

问题: {query}

请基于检索到的信息回答，如果信息不足请说明。"""

    result = model.invoke([{"role": "user", "content": prompt}])
    print(f"答案: {result.content}\n")


# ============================================================
# 第四部分：Agentic RAG（智能体式检索）
# ============================================================
# 智能体决定何时和如何检索
# 灵活性高，适用于需要访问多个工具的研究助理

@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索相关信息。传入搜索关键词。"""
    retrieved_docs = simple_retriever(query, SAMPLE_DOCUMENTS)
    if not retrieved_docs:
        return "未找到相关信息。"

    results = []
    for doc in retrieved_docs:
        results.append(f"[来源: {doc['source']}]\n{doc['content']}")

    return "\n\n".join(results)


def demo_agentic_rag():
    """演示 Agentic RAG。"""
    print("=" * 60)
    print("第四部分：Agentic RAG（智能体式检索）")
    print("=" * 60)

    agent = create_agent(
        model=create_model(temperature=0.3),
        tools=[search_knowledge_base],
        system_prompt="""你是一位知识助手，可以使用搜索工具从知识库中查找信息。

工作流程:
1. 分析用户问题，判断是否需要搜索知识库
2. 如果需要，使用 search_knowledge_base 工具搜索
3. 根据检索到的信息组织回答
4. 如果检索结果不足，可以多次搜索不同关键词

请确保回答基于检索到的信息，并引用来源。"""
    )

    print("测试1: 直接知识问题")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "LangChain 有哪些核心组件？"}]}
    )
    print(f"回答: {result['messages'][-1].content}\n")

    print("测试2: 需要多次搜索的问题")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "RAG 是什么？它和向量数据库有什么关系？"}]}
    )
    print(f"回答: {result['messages'][-1].content}\n")


# ============================================================
# 第五部分：三种 RAG 架构对比
# ============================================================

def demo_rag_comparison():
    """对比三种 RAG 架构。"""
    print("=" * 60)
    print("第五部分：三种 RAG 架构对比")
    print("=" * 60)

    print("""
| 架构        | 描述                              | 控制度 | 灵活性 | 延迟   | 适用场景             |
|-------------|-----------------------------------|--------|--------|--------|----------------------|
| 2 步 RAG    | 检索总在生成之前执行               | 高     | 低     | 快     | FAQ、文档机器人      |
| Agentic RAG | 智能体决定何时和如何检索            | 低     | 高     | 可变   | 研究助理、多工具场景  |
| 混合 RAG    | 结合两者特点，带查询增强和验证步骤  | 中     | 中     | 可变   | 带质量验证的领域问答  |

混合 RAG 的典型组件:
- 查询增强: 重写不清晰的查询、生成多个变体
- 检索验证: 评估检索到的文档是否相关和充分
- 答案验证: 检查答案的准确性和完整性
""")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("Demo 15: 检索 (Retrieval / RAG)\n")
    print("检索通过在查询时获取相关的外部知识，")
    print("解决 LLM 上下文有限和知识静态的问题。\n")

    # 构建知识库
    build_knowledge_base()

    # 演示 2 步 RAG
    demo_two_step_rag()

    # 演示 Agentic RAG
    demo_agentic_rag()

    # 架构对比
    demo_rag_comparison()

    print("\n检索总结:")
    print("1. 检索管道: 加载 -> 切分 -> 嵌入 -> 向量存储 -> 检索器")
    print("2. 2 步 RAG: 检索 -> 生成，简单可预测")
    print("3. Agentic RAG: 智能体决定何时检索，灵活但延迟可变")
    print("4. 混合 RAG: 结合查询增强、检索验证和答案验证")
    print("5. 核心组件: 文档加载器、切分器、嵌入模型、向量存储、检索器")
