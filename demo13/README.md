# Demo 13: 人机交互 (Human-in-the-Loop)

## 概述

HITL 中间件允许在代理工具调用中添加人工监督。当模型提出需要人工审查的操作时，中间件暂停执行并等待决策。

## 核心知识点

### 1. 中断决策类型

| 决策类型 | 描述 | 示例用例 |
|---|---|---|
| `approve` | 按原样批准并执行 | 完全按照草稿发送电子邮件 |
| `edit` | 修改后执行 | 在发送前更改收件人 |
| `reject` | 拒绝并添加反馈 | 拒绝邮件草稿并说明原因 |

### 2. 配置中断

通过 `HumanInTheLoopMiddleware` 的 `interrupt_on` 参数映射工具到允许的决策类型：

```python
interrupt_on={
    "write_file": True,  # 允许所有决策
    "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # 不允许编辑
    "read_data": False,  # 安全操作，无需批准
}
```

### 3. 执行生命周期

1. 代理调用模型生成响应
2. 中间件检查响应中的工具调用
3. 如果需要人工输入，构建 HITLRequest 并触发中断
4. 代理等待人工决策
5. 根据决策执行或拒绝，恢复执行

### 4. 响应中断

使用 `Command(resume={...})` 恢复暂停的对话：

```python
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config  # 相同的线程 ID
)
```

### 5. 必须配置 checkpointer

HITL 需要 checkpointer 来持久化中断时的图状态。测试用 `InMemorySaver()`，生产环境用 `AsyncPostgresSaver`。

## 文件说明

| 文件 | 说明 |
|---|---|
| `human_in_the_loop.py` | 完整示例代码 |

## 运行方法

```bash
python demo13/human_in_the_loop.py
```

## 参考文档

https://langchain-doc.cn/v1/python/langchain/human-in-the-loop.html
