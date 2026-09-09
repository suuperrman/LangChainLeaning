# Demo 9: 守卫 (Guardrails)

## 概述

为智能体实施安全检查和内容过滤。守卫通过在智能体执行的关键点验证和过滤内容，帮助构建安全、合规的 AI 应用。

## 核心知识点

### 1. 守卫的两种方法

| 方法 | 描述 |
|---|---|
| 确定性守卫 | 基于规则（正则、关键词匹配），快速、可预测、经济高效 |
| 基于模型的守卫 | 使用 LLM 评估内容，可捕获微妙问题，但速度较慢、成本较高 |

### 2. 内置守卫

- **PII 检测** (`PIIMiddleware`): 检测电子邮件、信用卡、IP 地址等，支持 redact/mask/hash/block 策略
- **人工审核** (`HumanInTheLoopMiddleware`): 在执行敏感操作前要求人工批准

### 3. 自定义守卫

- **智能体执行前** (`@before_agent`): 在每次调用开始时验证请求（身份验证、关键词过滤）
- **智能体执行后** (`@after_agent`): 在返回给用户之前验证最终输出（基于模型的安全检查）

### 4. 组合多个守卫

通过将多个守卫添加到 `middleware` 列表中堆叠使用，形成分层保护。

## 文件说明

| 文件 | 说明 |
|---|---|
| `guardrails.py` | 完整示例代码 |

## 运行方法

```bash
python demo9/guardrails.py
```

## 参考文档

https://langchain-doc.cn/v1/python/langchain/guardrails.html
