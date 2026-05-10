import json
import os
from pathlib import Path

import streamlit as st

from ir_system import InvertedIndex, Searcher, evaluate

st.set_page_config(page_title="中英信息检索系统", layout="wide")
st.title("📚 中英信息检索系统（带人工评测）")

INDEX_PATH = Path("data/index/index.json")
QRELS_PATH = Path("data/eval/qrels.json")

@st.cache_resource
def load_searcher():
    idx = InvertedIndex()
    idx.load("data/index")
    return Searcher(idx)


def ensure_qrels():
    if not QRELS_PATH.exists():
        QRELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        QRELS_PATH.write_text("{}", encoding="utf-8")


def read_qrels():
    ensure_qrels()
    return json.loads(QRELS_PATH.read_text(encoding="utf-8"))


def save_qrels(qrels):
    QRELS_PATH.write_text(json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8")


if not INDEX_PATH.exists():
    st.warning("未检测到索引文件 data/index/index.json，请先运行：python src/ir_system.py index")
    st.stop()

searcher = load_searcher()

with st.sidebar:
    st.header("检索参数")
    mode = st.selectbox("检索模型", ["bm25_prf", "bm25", "tfidf"], index=0)
    topk = st.slider("返回条数 TopK", 5, 30, 10)

query = st.text_input("请输入查询（中文或英文）", placeholder="例如：人工智能 / climate change")

if query:
    if mode == "bm25_prf":
        results = searcher.search_bm25_prf(query, topk=topk)
    elif mode == "bm25":
        results = searcher.search_bm25(query, topk=topk)
    else:
        results = searcher.search_tfidf(query, topk=topk)

    st.subheader(f"检索结果（{mode}）")
    qrels = read_qrels()
    selected = set(qrels.get(query, []))
    new_selected = set(selected)

    for i, (score, doc, snippet) in enumerate(results, start=1):
        with st.expander(f"#{i}  score={score:.4f}  |  {doc.title}"):
            st.write(f"**doc_id**: {doc.doc_id}")
            st.write(f"**date**: {doc.date}  |  **lang**: {doc.lang}")
            st.write(f"**url**: {doc.url}")
            st.write(f"**match**: {snippet}")
            checked = st.checkbox("标记为相关", value=(doc.doc_id in selected), key=f"{query}_{doc.doc_id}")
            if checked:
                new_selected.add(doc.doc_id)
            elif doc.doc_id in new_selected:
                new_selected.remove(doc.doc_id)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("保存当前查询标注"):
            qrels[query] = sorted(new_selected)
            save_qrels(qrels)
            st.success(f"已保存 {len(new_selected)} 条相关文档到 {QRELS_PATH}")
    with col2:
        if st.button("计算当前 qrels 指标"):
            try:
                # 复用原 evaluate 逻辑输出，避免重复实现
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    evaluate(str(QRELS_PATH), mode=mode)
                st.code(buf.getvalue())
            except Exception as e:
                st.error(f"评估失败: {e}")

st.markdown("---")
st.caption("启动方式：streamlit run src/ui_app.py")
