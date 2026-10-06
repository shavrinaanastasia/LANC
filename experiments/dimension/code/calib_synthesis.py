import pandas as pd, numpy as np, json
R="/home/claude/LANC/experiments/dimension/results/"
aux=json.load(open(R+"calib2_aux.json"))["meta"]
TD={k.split("_V")[0]:v["true_dim"] for k,v in aux.items() if k.endswith("V5000_T500")}
TD.update({"sphere2":2,"sphere6":6})
def cls(o):
    if o.startswith(("torus","sphere")): return "manifold"
    if o in ("lorenz","rossler","l96_6","l96_10"): return "flow"
    return "fractal"
rows=[]
a=pd.read_csv(R+"calib_table.csv")
for r in a.itertuples():
    rows.append(dict(src="E6",obj=r.obj,cond=f"part={r.part}",emb=r.emb,d=r.d,n=r.n,true=TD[r.obj],cls=cls(r.obj) if r.part!="kmeans" else "kmeans",
        twonn=r.twonn,schw=np.nanmean([r.schw_min,r.schw_max]) if not np.isnan(r.schw_min) else np.nan,hid=(r.hid_min+r.hid_max)/2,fishers=r.fishers))
b=pd.read_csv(R+"calib2_table.csv")
for r in b.itertuples():
    rows.append(dict(src="E10",obj=r.obj,cond=f"V={r.V},T={r.T}",emb=r.emb,d=r.d,n=r.n,true=TD[r.obj],cls=cls(r.obj),
        twonn=r.twonn,schw=np.nanmean([r.schw_min,r.schw_max]) if not np.isnan(r.schw_min) else np.nan,hid=(r.hid_min+r.hid_max)/2,fishers=r.fishers))
c=pd.read_csv(R+"calib3_table.csv")
for r in c.itertuples():
    rows.append(dict(src="E12",obj=r.obj,cond=f"step={r.step},k={r.kappa},V={r.vocab},s={r.sent},w={r.window}",emb=r.emb,d=r.d,n=r.n,true=r.m,cls="manifold",
        twonn=r.twonn,schw=r.schw,hid=r.hid,fishers=r.fishers))
p=pd.read_csv(R+"pseudo_table.csv")
for r in p.itertuples():
    e={"cbow_wap":"cbow","truth":"truth","svd":"svd"}.get(r.emb)
    if e is None: continue
    rows.append(dict(src="E7",obj=f"cube{r.m}",cond=f"walk={r.walk}",emb=e,d=r.d,n=r.n,true=r.m,cls="pseudo_eps" if r.walk=="eps" else "pseudo_dist",
        twonn=r.twonn,schw=np.nanmean([r.schw_min,r.schw_max]) if not np.isnan(r.schw_min) else np.nan,hid=(r.hid_min+r.hid_max)/2,fishers=r.fishers))
D=pd.DataFrame(rows)
for mth in ("twonn","schw","hid","fishers"): D["r_"+mth]=D[mth]/D.true
D.to_csv("pooled.csv",index=False)
print(D.groupby(["cls","emb"]).size().unstack(fill_value=0))
print(D.groupby("cls").obj.nunique())
import pandas as pd, numpy as np
from scipy.stats import spearmanr
D=pd.read_csv("pooled.csv")
D=D[~((D.emb!="truth")&(D.d==0))]
M=["twonn","schw","hid","fishers"]
def q(x): 
    x=x.dropna(); 
    return f"{np.median(x):.2f} [{np.quantile(x,.1):.2f}-{np.quantile(x,.9):.2f}] min{x.min():.2f} max{x.max():.2f} n={len(x)}" if len(x) else "-"
BR=D[D.cls.isin(["manifold","pseudo_eps"])]
print("=== truth (method alone), ratio est/m, Brownian manifolds")
for n in (1,2):
    for mth in M: print(n,mth,q(BR[(BR.emb=="truth")&(BR.n==n)]["r_"+mth]))
print("=== CBOW by d, Brownian")
for n in (1,2):
  for mth in M:
    print(n,mth," | ".join(f"d{dd}: {q(BR[(BR.emb=='cbow')&(BR.n==n)&(BR.d==dd)]['r_'+mth])}" for dd in (5,10,15,30,50,100)))
print("=== CBOW d15+30 by class")
for cl in ["manifold","pseudo_eps","pseudo_dist","flow","fractal","kmeans"]:
  for n in (1,2):
    s=D[(D.cls==cl)&(D.emb=="cbow")&D.d.isin([15,30])&(D.n==n)]
    print(cl,n," | ".join(f"{m}: {q(s['r_'+m])}" for m in M[:3]), "objs",s.obj.nunique())
print("=== dependence on m (Brownian, CBOW d15/30): spearman ratio vs m; median ratio per m")
for n in (1,2):
  s=BR[(BR.emb=="cbow")&BR.d.isin([15,30])&(BR.n==n)]
  for mth in M[:3]:
    t=s[["true","r_"+mth]].dropna(); rho=spearmanr(t.true,t["r_"+mth])[0]
    print(n,mth,f"rho={rho:.2f}", t.groupby("true")["r_"+mth].median().round(2).to_dict())
print("=== Schweinhart failures (no admissible alpha), CBOW d15/30")
s=D[(D.emb=="cbow")&D.d.isin([15,30])]
print(s.assign(f=s.schw.isna()).groupby(["cls","n"]).f.mean().round(2).unstack())
# LOO by object
print("=== LOO by object: relative error of m_hat (Brownian, CBOW d15/30)")
for n in (1,2):
  s=BR[(BR.emb=="cbow")&BR.d.isin([15,30])&(BR.n==n)]
  for mth in M[:3]:
    t=s[["obj","true",mth]].dropna(); errs_raw=[];errs_c=[];errs_l=[]
    for o in t.obj.unique():
        tr=t[t.obj!=o]; te=t[t.obj==o]
        f=np.median(tr[mth]/tr.true)
        b,a=np.polyfit(np.log(tr.true),np.log(tr[mth]),1)
        errs_raw+=list(te[mth]/te.true-1)
        errs_c+=list((te[mth]/f)/te.true-1)
        errs_l+=list(np.exp((np.log(te[mth])-a)/b)/te.true-1)
    f=lambda e: f"median|err| {np.median(np.abs(e))*100:.0f}%, 90% |err|<{np.quantile(np.abs(e),.9)*100:.0f}%, max {np.max(np.abs(e))*100:.0f}%"
    print(n,mth,"RAW:",f(errs_raw)," | CONST:",f(errs_c)," | LOGLOG:",f(errs_l), "objs",t.obj.nunique(),"configs",len(t))
import pandas as pd, numpy as np
D=pd.read_csv("pooled.csv")
BR=D[D.cls.isin(["manifold","pseudo_eps"])&(D.emb=="cbow")&D.d.isin([15,30])].copy()
def fam(r):
    if r.src=="E7": return "pseudo_eps"
    if r.src=="E6": return "E6 "+r.cond
    if r.src=="E10":
        return "E10 base" if r.cond=="V=5000,T=500" else ("E10 V-grid" if "T=500" in r.cond else "E10 T-grid")
    c=dict(x.split("=") for x in r.cond.split(","))
    if float(c["step"])!=1: return "E12 step"
    if float(c["k"])!=0: return "E12 zipf"+c["k"]
    if c["V"]!="5000": return "E12 V18000"
    if c["s"]!="20" or c["w"]!="5": return "E12 sent/window"
    return "E12 base"
BR["fam"]=BR.apply(fam,axis=1)
print(BR.groupby("fam").obj.nunique())
rng=np.random.default_rng(0)
for n in (1,2):
  for mth in ("twonn","schw","hid"):
    t=BR[BR.n==n][["fam","obj","true",mth]].dropna()
    out=[]
    for f in sorted(t.fam.unique()):
        tr=t[t.fam!=f]; te=t[t.fam==f]
        b,a=np.polyfit(np.log(tr.true),np.log(tr[mth]),1)
        e=np.exp((np.log(te[mth])-a)/b)/te.true-1
        out.append(f"{f}: {np.median(e)*100:+.0f}% (|e|max {np.abs(e).max()*100:.0f}%)")
    # bootstrap over objects for median ratio
    objs=t.obj.unique(); meds=[]
    for _ in range(2000):
        s=rng.choice(objs,len(objs)); meds.append(np.median(np.concatenate([(t[t.obj==o][mth]/t[t.obj==o].true).values for o in s])))
    print(f"\nn={n} {mth}: median ratio 95% CI over objects {np.quantile(meds,.025):.2f}-{np.quantile(meds,.975):.2f}")
    print("  leave-condition-family-out bias of m_hat:", " | ".join(out))
