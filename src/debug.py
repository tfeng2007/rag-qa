from rag import search

q = "偏差和方差分别对应欠拟合还是过拟合？模型复杂度如何影响它们？"
for i, (chunk, dist) in enumerate(search(q, top_k=3), 1):
    print(f"===== 第{i}块 (距离 {dist:.3f}) =====")
    print(chunk)          # 打印完整块，不截断
    print()