import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, json
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
t = pd.read_csv("calib2_table.csv"); a = json.load(open("calib2_aux.json")); M = a["meta"]; F = a["fnn"]
A = t[(t.V == 5000) & (t["T"] == 500)]
D = [5, 10, 15, 30, 50, 100]
def g(o, e, d, n, c="twonn"):
    v = A[(A.obj == o) & (A.emb == e) & (A.d == d) & (A.n == n)][c].values
    return v[0] if len(v) else np.nan
# fig 1: TwoNN vs d, words and bigrams, CBOW (solid) and SVD (dashed)
objs = [("torus2", "T²"), ("torus4", "T⁴"), ("torus6", "T⁶"), ("torus8", "T⁸"), ("torus10", "T¹⁰"), ("torus12", "T¹²"), ("l96_10", "Л-96 (6,55)"), ("lorenz", "Лоренц")]
cols = plt.cm.viridis(np.linspace(0, 0.9, len(objs)))
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
for ax, n, ttl in zip(axs, (1, 2), ("слова", "биграммы")):
    ax.plot([5, 100], [5, 100], "k--", lw=.6)
    for (o, lab), c in zip(objs, cols):
        ax.plot(D, [g(o, "cbow", d, n) for d in D], "o-", color=c, ms=3, label=lab)
        ax.plot(D, [g(o, "svd", d, n) for d in D], ":", color=c, lw=1)
    ax.set_xscale("log"); ax.set_xticks(D); ax.set_xticklabels(D); ax.set_ylim(0, 40); ax.grid(alpha=.3)
    ax.set_title(f"TwoNN, {ttl}: CBOW (сплошные), SVD (пунктир)"); ax.set_xlabel("d")
axs[0].set_ylabel("оценка"); axs[1].legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig("e1_bigd.png", dpi=200)
# fig 2: FNN / false-in-original / slope vs k for scalar delay
fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.3))
for o, lab, c in [("lorenz", "Лоренц (2,06)", "#2f6db5"), ("rossler", "Рёсслер (2,01)", "#38a169"), ("l96_6", "Л-96, N=6 (4,04)", "#d9822b"),
                  ("l96_10", "Л-96, N=10 (6,55)", "#c0392b"), ("torus6", "T⁶, броун. (6)", "#777777")]:
    s = F[o + "_V5000_T500"]["scalar_delay"]; ks = [int(k) for k in s if k != "tau"][:14]
    axs[0].plot(ks, [s[str(k)]["fnn10"] for k in ks], "o-", ms=3, color=c, label=lab)
    axs[1].plot(ks, [s[str(k)]["fo"]["gt3"] for k in ks], "o-", ms=3, color=c)
    axs[2].plot(ks, [s[str(k)]["cs"]["0.01"] for k in ks], "o-", ms=3, color=c)
for ax, ttl in zip(axs, ("FNN (Кеннел, Rtol 10)", "доля ложных в исходном пр-ве", "наклон корр. интеграла, C=0,01")):
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("размерность вложения k"); ax.grid(alpha=.3)
axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("e2_fnn.png", dpi=200)
# fig 3: k-gram slope for words (truth centres vs CBOW 30)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.3), sharey=True)
for o, lab, c in [("lorenz", "Лоренц", "#2f6db5"), ("rossler", "Рёсслер", "#38a169"), ("l96_6", "Л-96 N=6", "#d9822b"), ("l96_10", "Л-96 N=10", "#c0392b"),
                  ("torus2", "T²", "#999999"), ("torus6", "T⁶", "#555555"), ("gasket", "салфетка", "#8e44ad")]:
    for ax, key in zip(axs, ("words_truth_d0", "words_cbow_d30")):
        s = F[o + "_V5000_T500"][key]; ks = [int(k) for k in s]
        ax.plot(ks, [s[str(k)]["cs"]["0.01"] for k in ks], "o-", ms=3, color=c, label=lab)
axs[0].set_title("k слов подряд, координаты центров", fontsize=9); axs[1].set_title("k слов подряд, векторы CBOW d=30", fontsize=9)
for ax in axs: ax.set_xlabel("k (длина n-граммы)"); ax.grid(alpha=.3)
axs[0].set_ylabel("наклон корр. интеграла"); axs[0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig("e3_kgram.png", dpi=200)
# fig 4: inverse model
P = pd.read_csv("features_pred.csv", index_col=0)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
for cls, mk, c in (("sto", "o", "#555555"), ("det", "^", "#2f6db5")):
    q = P[P.cls == cls]; ax.scatter(q.m, q.pred_tw30n2, marker=mk, color=c, label={"sto": "группа «броуновская» (E2≈1)", "det": "группа E2 < 0,9"}[cls])
ax.plot([1, 13], [1, 13], "k--", lw=.7); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1.5, 2, 3, 4, 6, 8, 12]); ax.set_xticklabels([1.5, 2, 3, 4, 6, 8, 12]); ax.set_yticks([1.5, 2, 3, 4, 6, 8, 12]); ax.set_yticklabels([1.5, 2, 3, 4, 6, 8, 12])
ax.set_xlabel("истинная размерность"); ax.set_ylabel("восстановлено вслепую (LOO)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("e4_inverse.png", dpi=200)
print("ok")
