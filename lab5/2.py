# -*- coding: utf-8 -*-
"""
实验：基于 Hugging Face Transformers 的本地大模型对话与输入日志

【实验目标】
1) 使用本地已下载的指令/聊天模型实现一个最简对话循环；
2) 每次用户输入时，将三类信息分别落盘为 .txt：
   a. 原始输入文本
   b. 分词结果（逐 token：index、token_id、token_str）
   c. 输入词向量（embedding，逐 token 一行）
3) 理解 chat template 与纯文本 prompt 的差异，掌握从模型输入嵌入层提取向量的方法。
4) 仅完成TODO部分即可

#### 特别注意
运行part 2的作业文件前请指定gpu编号，避免占用别的组的卡号，我们按照 组号%8 定义gpu编号
即，第一组使用卡1，第二组使用卡2，依次类推，第八组使用卡0，第九组使用卡1，依此类推
使用CUDA_VISIBLE_DEVICES定义环境的gpu编号，命令如下

CUDA_VISIBLE_DEVICES=7 python 2.py

其中7根据组号自行定义
"""

import os
import time
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ************ 仅此处挖空：请学生在此完成离线加载 **************
# ========= 配置 =========

model_name_or_path = r"/data/shenzm/llama3/Meta-Llama-3.1-8B-Instruct"
output_dir = "./zzx/chat_logs"         # 日志输出目录，xxx自命名避免覆盖小组同学
max_new_tokens = 256               # 每次生成的最长新token数
temperature = 0.7
top_p = 0.9
use_chat_template = True           # 大多数指令/聊天模型建议 True

os.makedirs(output_dir, exist_ok=True)

# ========= 设备 & 精度 =========
cuda_ok = torch.cuda.is_available()
if cuda_ok:
    torch_dtype = torch.float16
else:
    # CPU 下用 float32 更稳妥
    torch_dtype = torch.float32

# ========= 加载分词器 & 模型（本地） =========
# TODO 要求：
# 1) 加载 tokenizer 与模型
# 2) tokenizer 建议 use_fast=True
# 3) 模型 dtype 使用上方 torch_dtype
# 4) GPU 环境可 device_map="auto"，CPU 环境可 model.to("cpu")
#
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    use_fast=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    torch_dtype=torch_dtype,
    device_map="auto" if cuda_ok else None
)
if not cuda_ok:
    model = model.to("cpu")

model.eval()  # 切换为推理模式

# 服务器端已为各组配好环境，请在服务器端运行此脚本
# *******************************************************

# ========= 工具函数 =========
def timestamp_str():
    return time.strftime("%Y%m%d-%H%M%S")

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_tokens(path, token_ids, tokens):
    # 保存为三列：index \t token_id \t token_str
    with open(path, "w", encoding="utf-8") as f:
        f.write("# index\ttoken_id\ttoken\n")
        for i, (tid, tok) in enumerate(zip(token_ids, tokens)):
            f.write(f"{i}\t{tid}\t{tok}\n")

def save_embeddings(path, tokens, embeddings, float_fmt="{:.6f}"):
    """
    embeddings: torch.Tensor [seq_len, hidden_size] （已经搬到CPU）
    逐行：index \t token \t v1 v2 v3 ... vH
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# index\ttoken\tembedding_vector\n")
        for i, tok in enumerate(tokens):
            vec = embeddings[i].tolist()
            vec_str = " ".join(float_fmt.format(x) for x in vec)
            f.write(f"{i}\t{tok}\t{vec_str}\n")

def log_user_input(turn_idx, user_text):
    """
    对用户输入进行：
      - 保存原文 txt
      - tokenizer -> 保存 token ids + tokens
      - 取 input embedding -> 保存为 txt
    """
    base = os.path.join(output_dir, f"{turn_idx:04d}_{timestamp_str()}")

    # 1) 原文
    raw_path = base + "_text.txt"
    save_text(raw_path, user_text)

    # 2) tokenize（只对“用户原文”分词，不加模板符号）
    enc = tokenizer(
        user_text,
        add_special_tokens=False,
        return_tensors="pt"
    )
    token_ids = enc.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tok_path = base + "_tokens.txt"
    save_tokens(tok_path, token_ids, tokens)

    # 3) embeddings（词向量）
    with torch.no_grad():
        input_ids = enc.input_ids.to(model.device)
        embed_layer = model.get_input_embeddings()  # nn.Embedding
        embeds = embed_layer(input_ids)             # [1, seq_len, hidden]
        embeds = embeds[0].detach().cpu()          # [seq_len, hidden]

    emb_path = base + "_embeddings.txt"
    save_embeddings(emb_path, tokens, embeds)

    print(f"[Saved] {raw_path}")
    print(f"[Saved] {tok_path}")
    print(f"[Saved] {emb_path}")

def build_prompt_from_history(history):
    """
    使用 chat template（若支持）或简单拼接。
    history: list[{"role":"user"/"assistant", "content":str}]
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            print(f"[Warn] apply_chat_template 失败，回退到简单拼接: {e}")

    # 回退：简单拼接
    lines = []
    for m in history:
        role = m.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)

def chat_loop():
    print("==== 本地对话开始（输入 /exit 退出）====")
    history = []
    turn_idx = 1

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_text.lower() in ("/exit", "/quit"):
            print("Bye!")
            break

        # 先记录用户输入（原文 / tokens / embeddings）
        log_user_input(turn_idx, user_text)

        # 更新聊天历史
        history.append({"role": "user", "content": user_text})

        # 构造提示词
        prompt = build_prompt_from_history(history)

        # 推理
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
            )

        # 仅解码新增部分
        out_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        print(f"\nAssistant: {response}")
        history.append({"role": "assistant", "content": response})

        turn_idx += 1

if __name__ == "__main__":
    # 小提示：首次运行若显存/内存不足，可换更小模型或在 CPU 上运行（会更慢）
    chat_loop()
