"""Aggregate calib3 (E12) results: over-estimation factor = estimate / true dimension (bigrams: / 2m)."""
import json, glob, os, sys, csv
R = sys.argv[1] if len(sys.argv) > 1 else "calib3_results"
rows = []
for f in sorted(glob.glob(f"{R}/*_n[12].json")):
    r = json.load(open(f))
    tag = os.path.basename(f).rsplit("_", 3)[0]
    meta = json.load(open(f"{R}/{tag}_meta.json"))
    m = meta["true_dim"]; n = r["ngram"]; t = m * n
    sch = (r.get("schweinhart") or {}).get("median")
    hid = r.get("hidalgo") or {}
    hm = sum(a * b for a, b in zip(hid.get("d_k", []), hid.get("p_k", []))) if hid.get("d_k") else None
    rows.append(dict(obj=r["object"], m=m, step=r["step"], kappa=r["kappa"], vocab=r["vocab"], sent=r["sent"], window=r["window"],
                     emb=r["embedding"], d=r["d"], n=n, twonn=r["twonn"], schw=sch, fishers=r["fishers"], hid=hm,
                     twonn_all=r.get("twonn_all"), N_all=r.get("N_all"), f_twonn=(r["twonn"] or 0) / t, f_schw=(sch / t if sch else None),
                     f_hid=(hm / t if hm else None), zipf_slope=meta["zipf_slope"], top1=meta["freq_top1pct_share"], used=meta["used_words"]))
out = os.path.join(os.path.dirname(R.rstrip("/")) or ".", "calib3_table.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(out, len(rows))
