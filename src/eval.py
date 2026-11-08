# src/eval.py —— RAG 检索评估
import json
import argparse
from pathlib import Path

from rag import search                    # 复用 rag.py 的检索（导入即建好索引/连上Chroma）
from llm_client import client, MODEL        # judge 要调 Claude
from rag import search, answer              # 原来只有 search，加上 answer
BASE_DIR = Path(__file__).resolve().parent.parent   # eval.py 在 src/ → 上一级是 rag-qa/
EVAL_PATH = BASE_DIR / "eval_set.json"


def load_eval_set(path=EVAL_PATH):
    return json.load(open(path, encoding="utf-8"))


def eval_retrieval(eval_set, top_k=3, verbose=True):
    single_scores, multi_scores = [], []
    details = []                                       # 每条明细，存文件对比用

    for item in eval_set:
        if item["type"] == "out_of_scope":
            continue                                   # 越界题不算普通命中率

        hits = search(item["question"], top_k)
        joined = " ".join(c for c, _ in hits).lower()  # 拼检索块 + 统一小写

        if item["type"] == "single":
            score = 1.0 if item["must_contain"].lower() in joined else 0.0
            single_scores.append(score)

        elif item["type"] == "multi":
            kps = item["key_points"]
            if not kps:                                # 防除零
                continue
            covered = sum(1 for k in kps if k.lower() in joined)
            score = covered / len(kps)                 # 覆盖率
            multi_scores.append(score)
        else:
            continue

        details.append({
            "question": item["question"],
            "type": item["type"],
            "score": round(score, 2),
        })

        if verbose and score < 1.0:                    # 打印未满分案例，方便定位
            expected = item.get("must_contain") or item.get("key_points")
            print(f"❌ [{score:.2f}] {item['question']}")
            print(f"   期望含: {expected}")
            print(f"   检索到: {[c[:40] for c, _ in hits]}")

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    single_rate, multi_rate = avg(single_scores), avg(multi_scores)

    print("\n=== 检索 eval ===")
    print(f"single 命中率:    {single_rate:.0%}  ({len(single_scores)} 题)")
    print(f"multi 平均覆盖率: {multi_rate:.0%}  ({len(multi_scores)} 题)")

    return {"single_rate": single_rate, "multi_rate": multi_rate, "details": details}

def compare(label_a, label_b):
    a = json.load(open(BASE_DIR / f"eval_result_{label_a}.json", encoding="utf-8"))
    b = json.load(open(BASE_DIR / f"eval_result_{label_b}.json", encoding="utf-8"))
    print(f"{'题目':38} {label_a:>10} {label_b:>10}")
    for da, db in zip(a["details"], b["details"]):
        flag = "  ← 变了" if da["score"] != db["score"] else ""
        print(f"{da['question'][:36]:38} {da['score']:>10} {db['score']:>10}{flag}")
    print(f"\nsingle_rate: {a['single_rate']:.0%} → {b['single_rate']:.0%}")
    print(f"multi_rate:  {a['multi_rate']:.0%} → {b['multi_rate']:.0%}")


def _extract_json(text):
    """从裁判输出里抠出 JSON"""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"judge没返回JSON: {text[:200]}")
    return json.loads(text[start:end + 1])


def judge(item, candidate):
    """对一条题的待评答案打分，返回 {score, reason}"""
    qtype = item["type"]

    # 按题型给不同评分标准
    if qtype == "out_of_scope":
        rubric = ('这是一道"资料中没有答案"的题。\n'
                  '- 待评答案老实表示"资料未提及/无法回答" → score=1\n'
                  '- 编造了具体内容（哪怕听起来合理） → score=0')
    elif qtype == "multi":
        rubric = ('对照"应覆盖要点"，看待评答案覆盖了几个。\n'
                  'score = 覆盖数 / 总数，取 0~1 的小数（如 2/3 填 0.67）。')
    else:  # single
        rubric = '待评答案与标准答案意思一致 → score=1；错误或答非所问 → score=0。'

    # multi 题额外把关键点列给裁判
    extra = ""
    if qtype == "multi" and item.get("key_points"):
        extra = f"\n应覆盖要点：{item['key_points']}"

    system = ("你是严格的评分员，按给定标准给RAG答案打分。"
              '只输出JSON：{"score": 数字, "reason": "简短理由"}，不要多余文字。')
    user = (f"评分标准：\n{rubric}\n\n"
            f"问题：{item['question']}\n"
            f"标准答案：{item['reference']}{extra}\n"
            f"待评答案：{candidate}")

    m = client.messages.create(
        model=MODEL, max_tokens=300, temperature=0,      # 裁判要稳，temp=0
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return _extract_json(m.content[0].text)

def eval_answers(eval_set, top_k=3, verbose=True):
    single, multi, oos = [], [], []       # 三类分开收分
    details = []

    for item in eval_set:
        ans, _ = answer(item["question"], top_k)     # 调 RAG 生成答案
        result = judge(item, ans)                    # 裁判打分
        score = float(result["score"])

        {"single": single, "multi": multi,
         "out_of_scope": oos}[item["type"]].append(score)   # 按type归类

        details.append({
            "question": item["question"], "type": item["type"],
            "score": round(score, 2), "answer": ans,
            "reason": result.get("reason", ""),
        })

        if verbose and score < 1.0:                  # 打印扣分案例
            print(f"❌ [{score:.2f}] ({item['type']}) {item['question']}")
            print(f"   答案: {ans[:80]}...")
            print(f"   判分理由: {result.get('reason', '')}")

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    print("\n=== 答案 eval ===")
    print(f"single 准确率:   {avg(single):.0%}  ({len(single)}题)")
    print(f"multi  覆盖率:   {avg(multi):.0%}  ({len(multi)}题)")
    print(f"越界   拒答正确率:{avg(oos):.0%}  ({len(oos)}题)")

    return {"single_acc": avg(single), "multi_cov": avg(multi),
            "oos_refuse": avg(oos), "details": details}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--mode", choices=["retrieval", "answer"], default="retrieval")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
    elif args.mode == "answer":
        result = eval_answers(load_eval_set(),top_k=5)
        out = BASE_DIR / f"eval_answer_{args.label}.json"       # 注意前缀 eval_answer_
        json.dump(result, open(out, "w"), ensure_ascii=False, indent=2)
        print(f"\n结果已存 {out.name}")
    else:
        result = eval_retrieval(load_eval_set(), top_k=5)
        out = BASE_DIR / f"eval_result_{args.label}.json"
        json.dump(result, open(out, "w"), ensure_ascii=False, indent=2)
        print(f"\n结果已存 {out.name}")