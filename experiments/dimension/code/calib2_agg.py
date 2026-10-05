import json, glob, os
rows = []
for f in sorted(glob.glob("calib2_results/*_n[12].json")):
    r = json.load(open(f)); sw = r.get("schweinhart") or {}; h = (r.get("hidalgo") or {}).get("d_k") or [None, None]
    rows.append([r["object"], r["vocab"], r["n_traj"], r["embedding"], r["d"], r["ngram"], r["N"], sw.get("min"), sw.get("max"),
                 r.get("twonn"), r.get("fishers"), min(h) if None not in h else None, max(h) if None not in h else None])
with open("calib2_table.csv", "w") as o:
    o.write("obj,V,T,emb,d,n,N,schw_min,schw_max,twonn,fishers,hid_min,hid_max\n")
    for x in rows:
        o.write(",".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in x) + "\n")
meta = {os.path.basename(f)[:-10]: json.load(open(f)) for f in glob.glob("calib2_results/*_meta.json")}
for m in meta.values(): m.pop("lyapunov", None)
fnn = {}
for f in glob.glob("calib2_results/*_fnn.json"):
    d = json.load(open(f)); out = {}
    for key, v in d.items():
        st = v["stats"] if "stats" in v else v
        if "error" in st: out[key] = st; continue
        out[key] = {k: {"fnn10": round(s["fnn"]["rtol10"], 4), "fnn2": round(s["fnn"]["rtol2"], 4), "atol": round(s["fnn"]["atol2"], 4),
                        "E1": s["E1"] and round(s["E1"], 4), "E2": s["E2"] and round(s["E2"], 4),
                        "tw": s["twonn_kcloud"] and round(s["twonn_kcloud"], 3),
                        "cs": {q: (c and round(c, 3)) for q, c in s.get("corr_slope", {}).items()},
                        "fo": s.get("false_orig")} for k, s in st.items()}
        if "tau" in v: out[key]["tau"] = v["tau"]
    fnn[os.path.basename(f)[:-9]] = out
json.dump({"meta": meta, "fnn": fnn}, open("calib2_aux.json", "w"))
print(len(rows), len(meta), len(fnn))
