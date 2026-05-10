import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple

import jieba
import requests
from bs4 import BeautifulSoup


ZH_STOP = {"的", "了", "和", "是", "在", "与", "及", "并", "或", "一个", "一种", "我们", "你", "我"}
EN_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "with", "and", "or", "is", "are", "be", "by"}


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    url: str
    date: str
    lang: str


class Crawler:
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
        return title, "\n".join(paragraphs[:12])

    def crawl_wikipedia(self, zh_count: int = 60, en_count: int = 60):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        docs = []
        targets = {"zh": zh_count, "en": en_count}
        got = {"zh": 0, "en": 0}
        visited = set()
        seeds = {
            "zh": "https://zh.wikipedia.org/wiki/Special:Random",
            "en": "https://en.wikipedia.org/wiki/Special:Random",
        }
        while got["zh"] < zh_count or got["en"] < en_count:
            for lang, seed in seeds.items():
                if got[lang] >= targets[lang]:
                    continue
                try:
                    resp = self.session.get(seed, timeout=20, allow_redirects=True)
                    url = resp.url
                    if url in visited:
                        continue
                    visited.add(url)
                    title, content = self._fetch_page(url)
                    if not content:
                        continue
                    docs.append(Document(f"{lang}_{got[lang]:04d}", title, content, url, datetime.utcnow().strftime("%Y-%m-%d"), lang))
                    got[lang] += 1
                    print(f"[{lang}] {got[lang]}/{targets[lang]}: {title}")
                    time.sleep(0.2)
                except Exception as e:
                    print(f"crawl error ({lang}): {e}")

        with open(self.out_path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
        print(f"Crawled {len(docs)} documents -> {self.out_path}")


class Tokenizer:
    @staticmethod
    def tokenize(text: str, lang: str) -> List[str]:
        text = text.lower()
        if lang == "zh":
            toks = [t.strip() for t in jieba.cut(text) if t.strip() and not re.match(r"^\W+$", t)]
            return [t for t in toks if t not in ZH_STOP and len(t) > 1]
        tokens = re.split(r"\s+", re.sub(r"[^a-z0-9 ]", " ", text))
        return [t for t in tokens if t and t not in EN_STOP and len(t) > 1]


class InvertedIndex:
    def __init__(self):
        self.postings = defaultdict(dict)
        self.doc_meta = {}
        self.doc_len = {}
        self.idf = {}
        self.doc_tf = {}
        self.doc_size = {}
        self.avgdl = 0.0

    def build(self, docs: List[Document], title_boost: float = 2.0):
        df = Counter()
        total_len = 0
        for d in docs:
            self.doc_meta[d.doc_id] = d
            tf_body = Counter(Tokenizer.tokenize(d.content, d.lang))
            tf_title = Counter(Tokenizer.tokenize(d.title, d.lang))
            tf = tf_body.copy()
            for t, c in tf_title.items():
                tf[t] += max(1, int(c * title_boost))

            self.doc_tf[d.doc_id] = tf
            self.doc_size[d.doc_id] = sum(tf.values())
            total_len += self.doc_size[d.doc_id]

            for term, freq in tf.items():
                self.postings[term][d.doc_id] = freq
            for term in tf.keys():
                df[term] += 1

        n_docs = len(docs)
        self.avgdl = total_len / max(n_docs, 1)
        for term, term_df in df.items():
            self.idf[term] = math.log((n_docs - term_df + 0.5) / (term_df + 0.5) + 1)

        for doc_id, tf in self.doc_tf.items():
            norm_sq = 0.0
            for term, freq in tf.items():
                w = (1 + math.log(freq)) * self.idf.get(term, 0.0)
                norm_sq += w * w
            self.doc_len[doc_id] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def save(self, out_dir: str = "data/index"):
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "postings": dict(self.postings),
            "doc_meta": {k: asdict(v) for k, v in self.doc_meta.items()},
            "doc_len": self.doc_len,
            "idf": self.idf,
            "doc_tf": {k: dict(v) for k, v in self.doc_tf.items()},
            "doc_size": self.doc_size,
            "avgdl": self.avgdl,
        }
        with open(f"{out_dir}/index.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, in_dir: str = "data/index"):
        with open(f"{in_dir}/index.json", "r", encoding="utf-8") as f:
            p = json.load(f)
        self.postings = defaultdict(dict, p["postings"])
        self.doc_meta = {k: Document(**v) for k, v in p["doc_meta"].items()}
        self.doc_len = p["doc_len"]
        self.idf = p["idf"]
        self.doc_tf = {k: Counter(v) for k, v in p["doc_tf"].items()}
        self.doc_size = {k: int(v) for k, v in p["doc_size"].items()}
        self.avgdl = float(p["avgdl"])


class Searcher:
    def __init__(self, index: InvertedIndex):
        self.index = index

    @staticmethod
    def _detect_lang(query: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"

    def _query_tokens(self, query: str):
        return Tokenizer.tokenize(query, self._detect_lang(query))

    def search_tfidf(self, query: str, topk: int = 10):
        q_tokens = self._query_tokens(query)
        q_tf = Counter(q_tokens)
        q_weights = {t: (1 + math.log(c)) * self.index.idf.get(t, 0.0) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        scores = defaultdict(float)
        for t, q_w in q_weights.items():
            idf = self.index.idf.get(t, 0.0)
            for doc_id, tf in self.index.postings.get(t, {}).items():
                scores[doc_id] += q_w * ((1 + math.log(tf)) * idf)
        return self._format(scores, q_norm, "tfidf", topk, q_tokens)

    def search_bm25(self, query: str, topk: int = 10, k1: float = 1.5, b: float = 0.75):
        q_tokens = self._query_tokens(query)
        q_tf = Counter(q_tokens)
        scores = defaultdict(float)
        for t, qf in q_tf.items():
            idf = self.index.idf.get(t, 0.0)
            for doc_id, tf in self.index.postings.get(t, {}).items():
                dl = self.index.doc_size.get(doc_id, 0)
                denom = tf + k1 * (1 - b + b * (dl / (self.index.avgdl or 1.0)))
                scores[doc_id] += idf * ((tf * (k1 + 1)) / (denom or 1.0)) * (1 + math.log(qf))
        return self._format(scores, 1.0, "bm25", topk, q_tokens)

    def _format(self, scores, q_norm, mode, topk, q_tokens):
        ranked = []
        for doc_id, s in scores.items():
            sim = s / ((q_norm * float(self.index.doc_len[doc_id])) if mode == "tfidf" else 1.0)
            d = self.index.doc_meta[doc_id]
            ranked.append((sim, d, self._snippet(d.content, q_tokens)))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:topk]

    @staticmethod
    def _snippet(text: str, tokens: List[str], window: int = 120) -> str:
        low = text.lower()
        pos = min([low.find(t.lower()) for t in tokens if t and low.find(t.lower()) >= 0] or [0])
        return text[max(0, pos - 20): max(0, pos - 20) + window].replace("\n", " ")


def load_docs(path="data/raw/documents.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]


def build_index(doc_path="data/raw/documents.jsonl"):
    idx = InvertedIndex()
    idx.build(load_docs(doc_path))
    idx.save("data/index")
    print("Index saved to data/index/index.json")


def interactive_search(mode="bm25"):
    idx = InvertedIndex()
    idx.load("data/index")
    s = Searcher(idx)
    print(f"IR 系统启动，检索模型: {mode}，输入 exit 退出")
    while True:
        q = input("query> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        results = s.search_bm25(q, 10) if mode == "bm25" else s.search_tfidf(q, 10)
        for i, (score, d, snip) in enumerate(results, 1):
            print(f"\n[{i}] score={score:.4f}")
            print(f"title: {d.title}")
            print(f"date: {d.date} | lang: {d.lang}")
            print(f"url: {d.url}")
            print(f"match: {snip}")


def evaluate(eval_file="data/eval/qrels.json", mode="bm25"):
    idx = InvertedIndex()
    idx.load("data/index")
    s = Searcher(idx)
    with open(eval_file, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    ap_list, p5_list = [], []
    for q, rel_docs in qrels.items():
        results = s.search_bm25(q, 20) if mode == "bm25" else s.search_tfidf(q, 20)
        ranked_ids = [d.doc_id for _, d, _ in results]
        rel = set(rel_docs)
        hit, precs = 0, []
        for rank, doc_id in enumerate(ranked_ids, 1):
            if doc_id in rel:
                hit += 1
                precs.append(hit / rank)
        ap_list.append(sum(precs) / max(len(rel), 1))
        p5_list.append(sum(1 for x in ranked_ids[:5] if x in rel) / 5)
    print(f"Mode: {mode}")
    print(f"MAP: {sum(ap_list)/len(ap_list):.4f}")
    print(f"P@5: {sum(p5_list)/len(p5_list):.4f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("crawl")
    c.add_argument("--zh", type=int, default=60)
    c.add_argument("--en", type=int, default=60)
    sub.add_parser("index")
    s = sub.add_parser("search")
    s.add_argument("--mode", choices=["bm25", "tfidf"], default="bm25")
    e = sub.add_parser("eval")
    e.add_argument("--file", default="data/eval/qrels.json")
    e.add_argument("--mode", choices=["bm25", "tfidf"], default="bm25")

    args = p.parse_args()
    if args.cmd == "crawl":
        Crawler().crawl_wikipedia(args.zh, args.en)
    elif args.cmd == "index":
        build_index()
    elif args.cmd == "search":
        interactive_search(args.mode)
    elif args.cmd == "eval":
        evaluate(args.file, args.mode)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
