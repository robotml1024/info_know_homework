import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple

import jieba
import requests
from bs4 import BeautifulSoup


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    url: str
    date: str
    lang: str


class Crawler:
    """从中英文维基页面抓取文档，自动抽取正文段落。"""

    def __init__(self, out_path: str = "data/raw/documents.jsonl"):
        self.out_path = out_path
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "InfoRetrievalHomeworkBot/1.0"})

    def _fetch_page(self, url: str) -> Tuple[str, str]:
        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("h1", {"id": "firstHeading"})
        title = title_tag.get_text(strip=True) if title_tag else ""

        content_div = soup.find("div", {"id": "mw-content-text"})
        paragraphs = []
        if content_div:
            for p in content_div.find_all("p"):
                txt = re.sub(r"\[\d+\]", "", p.get_text(" ", strip=True))
                if len(txt) > 40:
                    paragraphs.append(txt)
        content = "\n".join(paragraphs[:12])
        return title, content

    def crawl_wikipedia(self, zh_count: int = 60, en_count: int = 60):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        docs: List[Document] = []

        seeds = [
            ("zh", f"https://zh.wikipedia.org/wiki/Special:Random"),
            ("en", f"https://en.wikipedia.org/wiki/Special:Random"),
        ]
        target = {"zh": zh_count, "en": en_count}
        got = {"zh": 0, "en": 0}
        visited = set()

        while got["zh"] < zh_count or got["en"] < en_count:
            for lang, url in seeds:
                if got[lang] >= target[lang]:
                    continue
                try:
                    resp = self.session.get(url, timeout=20, allow_redirects=True)
                    final_url = resp.url
                    if final_url in visited:
                        continue
                    visited.add(final_url)

                    title, content = self._fetch_page(final_url)
                    if not content:
                        continue

                    doc = Document(
                        doc_id=f"{lang}_{got[lang]:04d}",
                        title=title,
                        content=content,
                        url=final_url,
                        date=datetime.utcnow().strftime("%Y-%m-%d"),
                        lang=lang,
                    )
                    docs.append(doc)
                    got[lang] += 1
                    print(f"[{lang}] {got[lang]}/{target[lang]}: {title}")
                    time.sleep(0.3)
                except Exception as e:
                    print(f"crawl error ({lang}): {e}")
                    continue

        with open(self.out_path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

        print(f"Crawled {len(docs)} documents -> {self.out_path}")


class Tokenizer:
    @staticmethod
    def tokenize(text: str, lang: str) -> List[str]:
        text = text.lower()
        if lang == "zh":
            return [t.strip() for t in jieba.cut(text) if t.strip() and not re.match(r"^\W+$", t)]
        tokens = re.split(r"\s+", re.sub(r"[^a-z0-9 ]", " ", text))
        return [t for t in tokens if t]


class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.doc_meta: Dict[str, Document] = {}
        self.doc_len: Dict[str, float] = {}
        self.idf: Dict[str, float] = {}

    def build(self, docs: List[Document]):
        df = Counter()
        tf_vectors = {}

        for d in docs:
            self.doc_meta[d.doc_id] = d
            tokens = Tokenizer.tokenize(d.title + " " + d.content, d.lang)
            tf = Counter(tokens)
            tf_vectors[d.doc_id] = tf
            for term, freq in tf.items():
                self.postings[term][d.doc_id] = freq
            for term in tf.keys():
                df[term] += 1

        n_docs = len(docs)
        for term, term_df in df.items():
            self.idf[term] = math.log((1 + n_docs) / (1 + term_df)) + 1

        for doc_id, tf in tf_vectors.items():
            norm_sq = 0.0
            for term, freq in tf.items():
                w = (1 + math.log(freq)) * self.idf[term]
                norm_sq += w * w
            self.doc_len[doc_id] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def save(self, out_dir: str = "data/index"):
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/postings.json", "w", encoding="utf-8") as f:
            json.dump(self.postings, f, ensure_ascii=False)
        with open(f"{out_dir}/doc_meta.json", "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.doc_meta.items()}, f, ensure_ascii=False)
        with open(f"{out_dir}/doc_len.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_len, f, ensure_ascii=False)
        with open(f"{out_dir}/idf.json", "w", encoding="utf-8") as f:
            json.dump(self.idf, f, ensure_ascii=False)
        print(f"Index saved to {out_dir}")

    def load(self, in_dir: str = "data/index"):
        with open(f"{in_dir}/postings.json", "r", encoding="utf-8") as f:
            self.postings = defaultdict(dict, json.load(f))
        with open(f"{in_dir}/doc_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.doc_meta = {k: Document(**v) for k, v in meta.items()}
        with open(f"{in_dir}/doc_len.json", "r", encoding="utf-8") as f:
            self.doc_len = json.load(f)
        with open(f"{in_dir}/idf.json", "r", encoding="utf-8") as f:
            self.idf = json.load(f)


class VectorSpaceSearcher:
    def __init__(self, index: InvertedIndex):
        self.index = index

    def _detect_lang(self, query: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"

    def search(self, query: str, topk: int = 10):
        lang = self._detect_lang(query)
        q_tokens = Tokenizer.tokenize(query, lang)
        q_tf = Counter(q_tokens)

        q_weights = {}
        for term, freq in q_tf.items():
            idf = self.index.idf.get(term, 0.0)
            q_weights[term] = (1 + math.log(freq)) * idf

        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        scores = defaultdict(float)

        for term, q_w in q_weights.items():
            postings = self.index.postings.get(term, {})
            idf = self.index.idf.get(term, 0.0)
            for doc_id, tf in postings.items():
                d_w = (1 + math.log(tf)) * idf
                scores[doc_id] += q_w * d_w

        ranked = []
        for doc_id, dot in scores.items():
            sim = dot / (q_norm * float(self.index.doc_len[doc_id]))
            d = self.index.doc_meta[doc_id]
            matched = self._extract_snippet(d.content, q_tokens)
            ranked.append((sim, d, matched))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:topk]

    @staticmethod
    def _extract_snippet(text: str, query_tokens: List[str], window: int = 120) -> str:
        pos = min([text.lower().find(t.lower()) for t in query_tokens if t and text.lower().find(t.lower()) >= 0] or [0])
        start = max(0, pos - 20)
        return text[start : start + window].replace("\n", " ")


def load_docs(path: str = "data/raw/documents.jsonl") -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(Document(**json.loads(line)))
    return docs


def build_index(doc_path: str = "data/raw/documents.jsonl"):
    docs = load_docs(doc_path)
    idx = InvertedIndex()
    idx.build(docs)
    idx.save("data/index")


def interactive_search():
    idx = InvertedIndex()
    idx.load("data/index")
    searcher = VectorSpaceSearcher(idx)

    print("IR 系统已启动，输入查询（exit 退出）：")
    while True:
        q = input("query> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        results = searcher.search(q, topk=10)
        for i, (score, doc, snippet) in enumerate(results, start=1):
            print(f"\n[{i}] score={score:.4f}")
            print(f"title: {doc.title}")
            print(f"date : {doc.date} | lang: {doc.lang}")
            print(f"url  : {doc.url}")
            print(f"match: {snippet}")


def evaluate(eval_file: str = "data/eval/qrels.json"):
    idx = InvertedIndex()
    idx.load("data/index")
    searcher = VectorSpaceSearcher(idx)

    with open(eval_file, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    ap_list = []
    p_at_5 = []
    for query, rel_docs in qrels.items():
        results = searcher.search(query, topk=20)
        ranked_ids = [d.doc_id for _, d, _ in results]
        rel_set = set(rel_docs)

        hit = 0
        precisions = []
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in rel_set:
                hit += 1
                precisions.append(hit / rank)
        ap = sum(precisions) / max(len(rel_set), 1)
        ap_list.append(ap)

        top5 = ranked_ids[:5]
        p5 = sum(1 for x in top5 if x in rel_set) / 5
        p_at_5.append(p5)

    print(f"MAP: {sum(ap_list)/len(ap_list):.4f}")
    print(f"P@5: {sum(p_at_5)/len(p_at_5):.4f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    crawl_p = sub.add_parser("crawl")
    crawl_p.add_argument("--zh", type=int, default=60)
    crawl_p.add_argument("--en", type=int, default=60)

    sub.add_parser("index")
    sub.add_parser("search")
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--file", default="data/eval/qrels.json")

    args = parser.parse_args()

    if args.cmd == "crawl":
        Crawler().crawl_wikipedia(args.zh, args.en)
    elif args.cmd == "index":
        build_index()
    elif args.cmd == "search":
        interactive_search()
    elif args.cmd == "eval":
        evaluate(args.file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
