下面是一版**精简版实验说明**，直接发给同学即可。

---

# 实验说明（精简版）

## 任务概览

你将完成一个“指令文本八分类”系统的**单线程**实现，并用老师给好的**多线程**脚本对你的单线程函数做并发加速验证。

* 你需要完成的文件：`dolly_tag_worker.py`（**已挖空**）
* 已写好的文件：`dolly_tag_mt.py`（**不用改**，跑通即可）
* 分类标签集合（去重后使用）：
  `["open_qa","creative_writing","information_extraction","classification","closeqa","brainstorming","general_qa"]`

## 单线程代码要做什么（目标说清楚）

请在 `dolly_tag_worker.py` 中**补全**以下功能，使其可以**单线程**完成整套流程：

1. **读取数据**
   从命令行参数 `--in_jsonl` 指定的 JSONL 文件读取样本。每行至少包含：

   * `instruction`（必填，字符串）
   * `context`（可选，字符串）
     可选字段：`id`、`response`（不参与打标，但一并透传）。

2. **调用模型做单条分类**

   * 组装 `messages = [system, user]`，其中：

     * `system` 放入已给出的分类说明 `SYSTEM_PROMPT`
     * `user` 把 `instruction` 与 `context` 直接拼接（有 `context` 时用空行分隔）
   * 使用百炼的api调用 `chat.completions.create`
   * 解析模型返回的 JSON，读取 `tag` 和 `reason`
   * 归一化 `tag`（小写、连字符转下划线）；若不在集合内则按关键词兜底到最接近的类，否则用 `general_qa`
   * 对单条调用做**最多 4 次重试**（指数退避），最终仍失败则返回 `("general_qa", "model_error: ...")`

3. **单线程遍历与汇总**

   * 逐条处理输入样本，得到每条记录的 `tag` 与 `tag_reason`
   * 组装结果字典，列表返回（注意保留原 `id`、`instruction`、`context`、`dolly_category`）

4. **写出结果**

   * 将结果保存为 `tag_results.jsonl`（逐行 JSON）和 `tag_results.csv`（方便表格查看）
   * 打印一个简单的标签分布统计

> 注意：`dolly_tag_mt.py` 会**直接调用你实现的 `tag_records(records, model, show_progress=False)`**。只要你的单线程实现正确，多线程脚本就能无缝并发跑起来。

## 多线程脚本怎么用（不用改）

* 老师已写好 `dolly_tag_mt.py`，它会把数据**平均分片**，用线程池**并发调用**你写好的 `tag_records`，合并结果并落盘。
* 你的任务是：先把**单线程**补全并验证能跑通，然后直接运行**多线程**脚本做加速测试即可。

## 环境准备

```bash
pip install openai==1.* pandas tqdm
export DASHSCOPE_API_KEY=sk-xxx
```

（也可在代码里直接写 key，但**不推荐**）

## 运行方法

1. 先跑**单线程**（你补全的脚本）：

```bash
python dolly_tag_worker.py --in_jsonl ./your_dataset.jsonl --model qwen-plus --n 200
```

* 期望输出：`tag_results.jsonl`、`tag_results.csv`，终端打印各标签数量。

2. 再跑**多线程**（老师已写好）：

```bash
python dolly_tag_mt.py --in_jsonl ./your_dataset.jsonl --model qwen-plus --n 200 --threads 6
```

* 期望输出：`tag_results_mt.jsonl`、`tag_results_mt.csv`，终端打印各标签数量。

## JSONL 输入示例

每行一个 JSON 对象：

```json
{"id": 0, "instruction": "请根据下文抽取公司名与成立时间。", "context": "阿里巴巴成立于1999年……"}
{"id": 1, "instruction": "写一首关于秋天的现代诗。", "context": ""}
```

## 验收标准（简版）

* 单线程脚本能**独立完成**读取→调用→解析→落盘→统计（核心）。
* 多线程脚本能**直接调用你的 `tag_records` 并成功跑完**（验证你接口设计正确）。
* 输出文件存在且格式合理；分类结果基本符合直觉（可抽检）。

就这些～先把**单线程**写对，再用**多线程**试一把加速。祝你实验顺利！
