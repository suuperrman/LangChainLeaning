# Demo 16: 长期记忆 (Long-term Memory)

## 概述

LangChain 智能体使用 LangGraph 持久化实现长期记忆。记忆存储在 Store 中，组织在 namespace 和 key 下。

## 核心知识点

### 1. 内存存储 (Memory Storage)

`InMemoryStore` 将数据保存到内存字典中（生产环境用数据库存储）：
- 每条记忆组织在 **namespace**（类似文件夹）和 **key**（类似文件名）下
- 支持跨命名空间搜索（基于内容过滤和向量相似度）

```python
store = InMemoryStore()
store.put(namespace, key, data)    # 写入
store.get(namespace, key)          # 读取
store.search(namespace, query=...) # 搜索
```

### 2. 在工具中读取长期记忆

通过 `runtime.store` 在工具内访问 Store：

```python
@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value)
```

### 3. 从工具中写入长期记忆

通过 `runtime.store.put()` 在工具内写入 Store：

```python
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    store.put(("users",), user_id, user_info)
    return "已保存"
```

### 4. 短期记忆 vs 长期记忆

| 类型 | 机制 | 范围 | 用途 |
|---|---|---|---|
| 短期记忆 | checkpointer (InMemorySaver) | 会话内 | 当前消息、工具结果 |
| 长期记忆 | store (InMemoryStore) | 跨会话 | 用户偏好、历史数据 |

## 文件说明

| 文件 | 说明 |
|---|---|
| `long_term_memory.py` | 完整示例代码 |

## 运行方法

```bash
python demo16/long_term_memory.py
```

## 参考文档

https://langchain-doc.cn/v1/python/langchain/long-term-memory.html
