# Demo 10: 运行时 (Runtime)

## 概述

LangChain 的 `create_agent` 在 LangGraph 的运行时环境下运行。Runtime 对象包含 Context、Store 和 Stream writer。

## 核心知识点

### 1. Runtime 的三大组成部分

| 组件 | 说明 | 范围 |
|---|---|---|
| Context (上下文) | 静态信息（用户 ID、数据库连接等） | 会话范围 |
| Store (存储) | `BaseStore` 实例，用于长期记忆 | 跨会话 |
| Stream writer (流写入器) | 通过 `"custom"` 流模式进行信息流式传输 | 运行时 |

### 2. 在工具内部访问运行时

使用 `ToolRuntime[Context]` 参数访问：
- `runtime.context`: 获取上下文
- `runtime.store`: 获取存储
- `runtime.writer`: 获取流写入器

### 3. 在中间件内部访问运行时

使用 `request.runtime` 访问：
- `@dynamic_prompt`: 根据运行时上下文动态生成系统提示
- `@before_model`: 模型调用前的钩子
- `@after_model`: 模型调用后的钩子

### 4. 使用 context_schema 定义上下文结构

通过 `@dataclass` 定义 Context，传入 `create_agent` 的 `context_schema` 参数。

## 文件说明

| 文件 | 说明 |
|---|---|
| `runtime.py` | 完整示例代码 |

## 运行方法

```bash
python demo10/runtime.py
```

## 参考文档

https://langchain-doc.cn/v1/python/langchain/runtime.html
