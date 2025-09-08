# =========================================================
# 实验名称：基于 DashScope 兼容接口的终端聊天（基础版）
# 面向对象：大二学生（已学 Python 基础 I/O）
#
# 实验目标：
# 1) 学会使用 百炼的api 调用对话模型（如 qwen-plus）。
# 2) 掌握最基本的对话循环：读取输入 → 调用接口 → 输出答案 → 维护历史 messages。
# 3) 能正确处理用户退出命令与异常（网络、鉴权失败等）。
#
# 任务与评分建议（可按需调整）：
# - 必做 (60%)：
#   A. 正确初始化客户端（api_key 与 base_url）；
#   B. 维护 messages（含一条 system 提示词）；
#   C. 循环读取用户输入并调用接口，打印模型回复；
#   D. 支持 /exit 退出；异常时不崩溃。
# - 进阶 (40%)：
#   E. 新增 /reset 指令（清空对话但保留 system）；
#   F. 将对话历史追加保存为 JSONL（/save chat_xxx.jsonl）。
#
# 运行前准备：
#   pip install openai==1.*
#   export DASHSCOPE_API_KEY=sk-xxx
#
# 运行示例：
#   python chat_simple.py
#
# 注意：
# - 本文件已在关键位置“挖空”。请按中文注释补全代码后再运行。
# =========================================================

import os
from openai import OpenAI

# TODO: 1) 初始化 OpenAI 客户端（DashScope 兼容）
#   - 使用环境变量 DASHSCOPE_API_KEY 或直接写入 api_key="sk-xxx"
#   - base_url 必须是 "https://dashscope.aliyuncs.com/compatible-mode/v1"
#   - 参考：
#       client = OpenAI(api_key=..., base_url=...)
client = OpenAI(
    api_key="sk-92ea8daa134845369692e52bffe21f09",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# TODO: 2) 选择模型名（可从环境变量 MODEL 读取，默认 "qwen-plus"）
MODEL = os.getenv("MODEL", "qwen-plus")

def main():
    # TODO: 3) 初始化 messages，并包含一条 system 提示词
    #   messages 示例：[{"role": "system", "content": "You are a helpful assistant."}]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    # TODO: 4) 打印提示信息，例如“已就绪，输入 /exit 退出”
    print("聊天机器人已就绪！输入 /exit 退出，输入 /reset 重置对话，输入 /save 保存对话历史")
    # print(...)

    while True:
        try:
            # TODO: 5) 从终端读取一行用户输入（去掉首尾空格）
            # q = input(...).strip()
            q = input("用户：").strip()
        except (KeyboardInterrupt, EOFError):
            # TODO: 6) 处理 Ctrl+C / EOF 退出：打印提示后 break
            print("\n再见！")
            break

        # TODO: 7) 忽略空输入（continue），支持 '/exit'/'exit'/'quit' 主动退出
        # if not q: ...
        # if q.lower() in ("/exit", "exit", "quit"): ...
        if not q:
            continue
            
        if q.lower() in ("/exit", "exit", "quit"):
            print("再见！")
            break

        # 处理 /reset 命令
        if q.lower() == "/reset":
            # 保留系统提示，清空其他消息
            system_msg = messages[0] if messages and messages[0]["role"] == "system" else {"role": "system", "content": "You are a helpful assistant."}
            messages = [system_msg]
            print("对话历史已重置")
            continue
            
        # 处理 /save 命令
        if q.lower() == "/save":
            # 生成带时间戳的文件名
            timestamp = int(time.time())
            filename = f"chat_{timestamp}.jsonl"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    for msg in messages:
                        # 跳过系统消息
                        if msg["role"] != "system":
                            f.write(json.dumps(msg, ensure_ascii=False) + '\n')
                print(f"对话历史已保存到 {filename}")
            except Exception as e:
                print(f"保存失败：{e}")
            continue

        # TODO: 8) 将用户输入追加到 messages（role="user"）
        # messages.append({...})
        messages.append({"role": "user", "content": q})

        try:
            # TODO: 9) 发起一次 Chat Completion 调用
            # resp = client.chat.completions.create(model=MODEL, messages=messages)
            resp = client.chat.completions.create(model=MODEL, messages=messages)
            # TODO: 10) 从 resp 中取出 assistant 文本
            # a = resp.choices[0].message.content
            a = resp.choices[0].message.content
        except Exception as e:
            # TODO: 11) 调用失败时打印错误，并回滚刚刚追加的 user 消息
            print(f"调用失败：{e}")
            messages.pop()
            continue

        # TODO: 12) 打印 assistant 文本到终端
        # print(f"助理：{a}")
        print(f"助理：{a}")

        # TODO: 13) 将 assistant 回复也追加进 messages（role="assistant"）
        # messages.append({...})
        messages.append({"role": "assistant", "content": a})

if __name__ == "__main__":
    main()
