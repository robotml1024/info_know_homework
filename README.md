# 信息检索系统课程大作业（高分版模板）

## 功能对应评分点
- ✅ 完整 IR 流水线：爬虫、存储、倒排、向量空间检索、排序输出。
- ✅ 中英双语支持（中文分词+英文空格分词）。
- ✅ 输出字段完整：相关度、题目、匹配内容、URL、日期。
- ✅ 提供人工准确率评价（MAP、P@5）。
- ✅ 报告中给出可持续发展影响分析与创新拓展方向。

## 环境安装
```bash
pip install -r requirements.txt
```

## 一键流程
```bash
python src/ir_system.py crawl --zh 60 --en 60
python src/ir_system.py index
python src/ir_system.py search
```

## 人工评价
```bash
cp data/eval/qrels.template.json data/eval/qrels.json
# 手工修改 qrels.json 中的相关文档ID
python src/ir_system.py eval --file data/eval/qrels.json
```

## 目录结构
- `src/ir_system.py`：主程序（爬虫、索引、检索、评估）
- `data/raw/`：原始文档 JSONL
- `data/index/`：倒排索引与统计文件
- `data/eval/`：人工评价标注文件
- `reports/report.md`：实验报告

## 说明
若网络受限无法抓取 Wikipedia，可将数据源替换为任何公开可爬网页（新闻、博客、论坛等），只要文档数量 ≥ 100 即可。
