# 史记智能体 (Shiji Agent)

基于开源知识图谱 [baojie/shiji-kb](https://github.com/baojie/shiji-kb)（57万字《史记》→ 14,065实体 / 3,198事件 / 130篇时间线）的问答智能体。实体/纪年/事件走确定性查表，语义推理走 LLM（DeepSeek），全文检索为纯 stdlib bigram TF-IDF —— **零第三方依赖，仅 Python 标准库**。

## 运行

```bash
python shiji_agent.py "项羽和刘邦是什么关系?"   # 命令行问答
python shiji_agent.py --web                    # Web 界面 http://127.0.0.1:5210
python shiji_agent.py --selfcheck              # 6 项自检
python shiji_agent.py --demo                   # 自检 + 2 个验收问答
```

API key：环境变量 `DEEPSEEK_API_KEY` 优先，或项目根 `deepseek_key.tmp`（已 gitignore，不入库）。

## 数据说明

本仓**不附带知识图谱数据**：首次运行时 `shiji-kb-data/`（11MB）与 `chapter_cache/`（130 章标注语料）会自动从 ghproxy / ghfast 镜像拉取 `baojie/shiji-kb` 并缓存到本地。

数据版权归 [baojie/shiji-kb](https://github.com/baojie/shiji-kb)（鲍捷, 2026）所有，协议 **CC BY-NC-SA 4.0**（署名-非商业-相同方式共享）。《史记》原文属公有领域，受限的是其标注/图谱层——运行时拉取缓存不构成本仓的再分发。

## 相关项目

[江山时序 jiangshan-map](https://github.com/DeanChensj/jiangshan-map)（GPL-3.0）— 2400 年中国历史交互地图，可作本智能体的时空参照。如需本地部署：`git clone https://github.com/DeanChensj/jiangshan-map.git`。

## License

代码 MIT；数据部分遵循上游 CC BY-NC-SA 4.0。
