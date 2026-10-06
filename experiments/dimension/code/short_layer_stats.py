"""E9: per-layer text statistics of the short layer sweep (words per dialogue, repeated utterances, trigram repetition,
type-token ratio, empty utterances). Run on HSE: python3 short_layer_stats.py -> results/short_layer_stats.json"""
import json, glob, re, collections, statistics as st
R = "/home/aoshavrina/lanc_run/results/layer-sweep-pca32-v1/short_shards/*.jsonl"
acc = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(R):
    for line in open(f):
        r = json.loads(line)
        if r.get("status") != "complete":
            continue
        L = r["intervention"].split("_")[0]; fr = r["interaction_frame"]
        toks = [re.findall(r"\w+", (u["text"] or "").lower()) for u in r["utterances"]]
        norm = [" ".join(t) for t in toks]; allw = [w for t in toks for w in t]
        tri = [tuple(allw[i:i + 3]) for i in range(len(allw) - 2)]
        a = acc[(L, fr)]
        a["words"].append(len(allw))
        a["rep"].append(sum(1 for i, s in enumerate(norm) if s and s in norm[:i]) / max(1, len(norm)))
        a["trirep"].append(1 - len(set(tri)) / len(tri) if tri else 0)
        a["ttr"].append(len(set(allw)) / len(allw) if allw else 0)
        a["empty"].append(sum(1 for s in norm if not s) / max(1, len(norm)))
out = [dict(layer=L, frame=fr, n=len(a["words"]), **{k: round(st.mean(v), 4) for k, v in a.items()}) for (L, fr), a in sorted(acc.items())]
json.dump(out, open("/home/aoshavrina/lanc_run/results/short_layer_stats.json", "w"), indent=0)
