# Demo 12: 模型上下文协议 (MCP)

## 概述

MCP (Model Context Protocol) 是一种开放协议，标准化了应用程序如何向 LLM 提供工具和上下文。LangChain 通过 `langchain-mcp-adapters` 库连接 MCP 服务器。

## 核心知识点

### 1. 传输类型

| 传输类型 | 说明 | 适用场景 |
|---|---|---|
| stdio | 客户端启动服务器子进程，通过标准 I/O 通信 | 本地工具、简单设置 |
| Streamable HTTP | 服务器独立运行，处理 HTTP 请求 | 远程连接、多客户端 |
| SSE | Streamable HTTP 的变体，优化实时流式通信 | 实时通信 |

### 2. 使用 MCP 工具

使用 `MultiServerMCPClient` 连接多个 MCP 服务器，获取工具后传给 `create_agent`。

### 3. 自定义 MCP 服务器

使用 `mcp` 库的 `FastMCP` 创建工具服务器：
- `math_server.py`: stdio 传输的数学服务器
- `weather_server.py`: Streamable HTTP 传输的天气服务器

### 4. 有状态 vs 无状态

- **无状态（默认）**: 每次工具调用创建新的 ClientSession
- **有状态**: 使用 `client.session()` 创建持久会话

## 安装依赖

```bash
pip install langchain-mcp-adapters mcp
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `mcp.py` | 完整示例代码（含 MCP 服务器代码模板） |

## 运行方法

```bash
# 1. 安装依赖
pip install langchain-mcp-adapters mcp

# 2. 创建并启动 MCP 服务器（代码在 mcp.py 中）
python math_server.py     # 终端1
python weather_server.py  # 终端2

# 3. 运行 demo
python demo12/mcp.py
```

## 参考文档

https://langchain-doc.cn/v1/python/langchain/mcp.html
