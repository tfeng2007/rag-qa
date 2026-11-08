import chromadb
import re
from sentence_transformers import SentenceTransformer
from llm_client import client, MODEL
from pathlib import Path
import argparse
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
BASE_DIR = Path(__file__).resolve().parent.parent   # src 的上一级 = rag-qa/
DATA_DIR = BASE_DIR / "data"


def chunk_text(text, size=300, overlap=50):
    """把长文本切成带重叠的小块"""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# META_HEADINGS = ("学习目标", "本章小结", "小结", "思考题", "下一章预告", "岔路口")

def chunk_by_header(text, max_size=1500, overlap=50):
    # 在每个标题行前切开（标题跟着它的内容留在一起）
    sections = re.split(r'\n(?=#{1,2}\s)', text)  # #{1,6} 改成 #{1,2}
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:                                    # 过滤空段
            continue

        # first_line = sec.split("\n", 1)[0]  # 取这节的标题行
        # if any(kw in first_line for kw in META_HEADINGS):  # 标题含元信息词就跳过
        #     continue

        if len(sec) <= max_size:
            chunks.append(sec)                         # 不太长，整节做一块
        else:
            chunks.extend(chunk_text(sec, max_size, overlap))  # 太长，兜底按字数切
    return chunks

# ---------- 用 Chroma 建索引 ----------
chroma = chromadb.PersistentClient(path="./chroma_db")      # 持久化到磁盘
collection = chroma.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},                       # 用余弦相似度
)

def build_index(docs):
    chunks = []
    for d in docs:
        chunks.extend(chunk_by_header(d))
    vecs = embedder.encode(chunks, normalize_embeddings=True).tolist()  # 转成list
    ids = [f"chunk_{i}" for i in range(len(chunks))]         # 每块要个唯一id
    collection.upsert(ids=ids, documents=chunks, embeddings=vecs)  # 存进Chroma

# ---------- 用 Chroma 检索 ----------
# def search(query, top_k=5):
#     q = embedder.encode([query], normalize_embeddings=True).tolist()
#     res = collection.query(query_embeddings=q, n_results=top_k)
#     docs = res["documents"][0]        # 命中的文本块
#     dists = res["distances"][0]       # 距离（越小越相似）
#     return list(zip(docs, dists))

def search(query, top_k=5, candidates=20):
    # 第①段：embedding 召回更多候选
    q = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q, n_results=candidates)
    docs = res["documents"][0]

    # 第②段：rerank 精排
    pairs = [(query, d) for d in docs]          # 组成 (问题,块) 对
    scores = reranker.predict(pairs)            # 每对算一个相关性分(越大越相关)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_k]]   # 取精排后 top_k

def answer(query, top_k=5):
    hits = search(query, top_k)
    context = "\n\n---\n\n".join(c for c, _ in hits)
    m = client.messages.create(
        model=MODEL, max_tokens=1024, temperature=0,
        system="只依据<资料>回答。资料没有就说「资料中未提及」，不要编造。",
        messages=[{"role": "user", "content": f"<资料>\n{context}\n</资料>\n\n问题：{query}"}],
    )
    return m.content[0].text, hits

def load_docs(folder=DATA_DIR):
    folder = Path(folder)
    docs = []
    for path in folder.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append(text)
        print(f"读入 {path.name}，{len(text)} 字")
    if not docs:                                      # 空了就明确报错，别拖到 Chroma
        raise FileNotFoundError(f"{folder} 里没找到任何 .md 文件，检查文件放对没")
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="删除旧索引并重建")
    args = parser.parse_args()

    global collection  # 因为下面要重新赋值这个模块级全局变量

    if args.rebuild:
        print("删除旧索引，重建中...")
        chroma.delete_collection(name="docs")  # 删掉旧的
        collection = chroma.get_or_create_collection(  # 建个新的空的
            name="docs", metadata={"hnsw:space": "cosine"},
        )
        build_index(load_docs())
    elif collection.count() == 0:
        print("首次建索引...")
        build_index(load_docs())
    else:
        print(f"已有 {collection.count()} 个块，跳过建索引")
    reply, hits = answer("决策树为什么要剪枝？")
    for c, d in hits:
        print(f"[距离 {d:.3f}] {c[:80]}...")
    print("\n回答：", reply)

if __name__ == "__main__":
    main()