from rag import embedder, collection      # 直接借 rag 里的 embedder 和 collection

query = "决策树有哪几种划分选择的度量？分别对应什么算法？"
q = embedder.encode([query], normalize_embeddings=True).tolist()
res = collection.query(query_embeddings=q, n_results=20)   # 召回20个，不经过rerank

print("=== embedding 召回的 top-20 候选（rerank 之前）===")
for i, d in enumerate(res["documents"][0], 1):
    print(f"{i:2}. {d[:45]}")              # 每块打印前45字，看标题