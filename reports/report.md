# 信息与知识获取大作业实验报告

## 1. 项目目标
实现一个中英双语信息检索系统，包含：网络爬虫、文档本地存储、倒排索引、向量空间模型检索、结果排序展示、人工准确率评估。

## 2. 系统设计
- 数据采集：爬取中英文 Wikipedia 随机词条。
- 规模：默认中文 60 + 英文 60，共 120 篇文档（可配置）。
- 存储：JSONL（每行一个文档）。
- 预处理：
  - 中文：`jieba` 分词；
  - 英文：空格与正则标准化分词。
- 索引：倒排索引（term -> {doc_id: tf}），并保存 IDF、文档向量范数。
- 检索：TF-IDF + 余弦相似度（向量空间模型）。
- 输出字段：相关度 score、题目 title、匹配片段 snippet、URL、日期 date、语言 lang。

## 3. 核心算法
- `idf(t) = log((1+N)/(1+df(t))) + 1`
- `w(t,d) = (1 + log(tf(t,d))) * idf(t)`
- `sim(q,d) = (Σ w(t,q)w(t,d)) / (||q|| ||d||)`

## 4. 人工评价方法
1. 先运行检索系统，选取若干典型查询（至少 10 个）。
2. 人工标注每个查询的相关文档集合，写入 `data/eval/qrels.json`。
3. 运行评估脚本输出 MAP、P@5。

## 5. 可持续发展与社会影响
- 算法层：使用稀疏倒排与轻量 TF-IDF，降低算力消耗。
- 工程层：本地索引与增量复用，减少重复爬取与网络流量。
- 社会层：支持透明检索结果（URL、日期、片段）以提升信息可追溯性，降低错误信息传播风险。

## 6. 可扩展创新点
- BM25 替代 TF-IDF。
- 加入学习排序（LTR）特征。
- 多媒体检索：图文联合索引（CLIP embedding）。
- 查询扩展：同义词词林/WordNet。

## 7. 复现实验命令
```bash
python src/ir_system.py crawl --zh 60 --en 60
python src/ir_system.py index
python src/ir_system.py search
cp data/eval/qrels.template.json data/eval/qrels.json
python src/ir_system.py eval --file data/eval/qrels.json
```
