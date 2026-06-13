# -*- coding: utf-8 -*-
"""
m5_guarded_ensemble.py — V-TWIN framework, Model M5
Guarded selective classifier with conformal abstention.

Bileşenler:
  D : hastalık modeli  P(LC)   (Voice45 full)
  C : confound-only modeli P(LC | QC + AGE)  — "edinim/demografi ile açıklanabilirlik"
  Conformal (split LAC): hedef seçici-hata α için kalibre edilmiş prediction-set; singleton değilse abstain.
  Confound guard: confound modeli emin ise (|P_C-0.5| büyük) → karar edinimle açıklanabilir → abstain.

Çıktı: risk-coverage eğrileri + konformal işletme noktası; results/m5_*.csv + Figure5 PNG.
Gerçek IBNI verisi (LC vs NON_LC).
"""
import os, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import matplotlib.pyplot as plt

RS = 0
PATH = "features_local_ibni_candidate_subjectlevel.csv"   # gerçek IBNI (canonical voice45)
OUTFIG = "/mnt/user-data/outputs/figures/Figure5_guarded_ensemble.png"
RESDIR = "results"; os.makedirs(RESDIR, exist_ok=True)

FEAT = ([f"mfcc_{i}" for i in range(13)] + [f"delta_mfcc_{i}" for i in range(13)] +
        ["spectral_centroid_mean","spectral_flatness_mean","spectral_rolloff_mean",
         "spectral_flux_mean","rms_mean","zcr_mean","energy_mean"] +
        ["spectral_centroid_std","spectral_flatness_std","spectral_rolloff_std","rms_std","zcr_std"] +
        ["f0_mean","f0_std","jitter","shimmer","hnr","voiced_fraction","breathiness"])
QC = ["duration_sec","overall_dbfs","speech_dbfs","speech_frame_fraction","selected_segment_count"]
CONF = QC + ["AGE"]

def model(): return Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=2000,C=1.0))])

def oof_proba(X, y, reps=10):
    """tekrarlı stratified CV ortalama out-of-fold P(LC)."""
    y=np.asarray(y); acc=np.zeros(len(y))
    k=min(5,int(np.min(np.bincount(y))))
    for r in range(reps):
        skf=StratifiedKFold(k,shuffle=True,random_state=r)
        acc+=cross_val_predict(model(),X,y,cv=skf,method="predict_proba")[:,1]
    return acc/reps

def main():
    df=pd.read_csv(PATH)
    if "site_inferred" in df.columns and (df["site_inferred"]=="IBNI").any():
        df=df[df["site_inferred"]=="IBNI"].copy()
    df=df[df["label"].isin(["LC","NON_LC"])].copy()
    Xf=df[FEAT].apply(pd.to_numeric,errors="coerce"); keep=~Xf.isna().any(axis=1)
    df,Xf=df[keep],Xf[keep]
    y=(df["label"]=="LC").astype(int).values
    Xc=df[CONF].apply(pd.to_numeric,errors="coerce").fillna(df[CONF].median()).values

    P_D=oof_proba(Xf.values,y)              # disease score
    P_C=oof_proba(Xc,y)                     # confound-only score
    conf_D=2*np.abs(P_D-0.5)                 # 0..1
    conf_C=2*np.abs(P_C-0.5)
    yhat=(P_D>0.5).astype(int)
    correct=(yhat==y).astype(int)
    n=len(y); base_acc=correct.mean()
    print(f"n={n}  LC={y.sum()}  base accuracy(P_D)={base_acc:.3f}")

    # ---------- risk-coverage (selective) ----------
    def curve(order_mask=None):
        # keep subjects allowed by mask, then sort by conf_D desc; sweep coverage
        idx=np.arange(n) if order_mask is None else np.where(order_mask)[0]
        order=idx[np.argsort(-conf_D[idx])]
        covs=[]; accs=[]
        for kkeep in range(1,len(order)+1):
            sel=order[:kkeep]
            covs.append(len(sel)/n); accs.append(correct[sel].mean())
        return np.array(covs),np.array(accs)

    covA,accA=curve(None)                                   # disease-confidence selective
    GAMMA=0.5                                                # confound-guard eşiği (conf_C<GAMMA tut)
    guard_mask=conf_C<GAMMA
    covB,accB=curve(guard_mask)                             # guarded (confound-açıklanabilir olanları at)
    print(f"confound-guard: {guard_mask.mean()*100:.0f}% subjects pass guard (conf_C<{GAMMA})")

    # ---------- split-conformal LAC operating point ----------
    rng=np.random.default_rng(0); perm=rng.permutation(n); half=n//2
    cal,te=perm[:half],perm[half:]
    P=np.c_[1-P_D,P_D]                                       # [P(non), P(LC)]
    def conformal(alpha, extra_keep=None):
        s=1-P[cal,y[cal]]                                    # nonconformity on cal
        qlev=min(1.0,np.ceil((len(cal)+1)*(1-alpha))/len(cal))
        qhat=np.quantile(s,qlev,method="higher")
        sets=P[te]>=(1-qhat)                                 # boolean [n_te,2]
        singleton=sets.sum(1)==1
        if extra_keep is not None: singleton=singleton & extra_keep[te]
        pred=np.where(P[te,1]>=P[te,0],1,0)
        cov=singleton.mean()
        acc=(pred[singleton]==y[te][singleton]).mean() if singleton.sum() else np.nan
        return cov,acc
    rows=[]
    for a in [0.10,0.20,0.30]:
        c0,a0=conformal(a); c1,a1=conformal(a, guard_mask)
        rows.append(dict(alpha=a, cov_disease=c0, acc_disease=a0, cov_guarded=c1, acc_guarded=a1))
        print(f"α={a}: disease-only cov={c0:.2f} acc={a0:.3f} | guarded cov={c1:.2f} acc={a1:.3f}")
    pd.DataFrame(rows).to_csv(f"{RESDIR}/m5_conformal_operating_points.csv",index=False)
    pd.DataFrame(dict(coverage=covA,acc=accA)).to_csv(f"{RESDIR}/m5_riskcoverage_disease.csv",index=False)
    pd.DataFrame(dict(coverage=covB,acc=accB)).to_csv(f"{RESDIR}/m5_riskcoverage_guarded.csv",index=False)

    # ---------- figure ----------
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.plot(covA,accA,color="#2E5496",lw=2,label="Disease-confidence only")
    ax.plot(covB,accB,color="#B0413E",lw=2,label="Guarded (confound-aware)")
    ax.axhline(0.5,ls="--",lw=1,color="#999"); ax.axhline(base_acc,ls=":",lw=1,color="#2E5496")
    op=rows[1]  # α=0.20
    ax.scatter([op["cov_guarded"]],[op["acc_guarded"]],color="#B0413E",zorder=5,s=40)
    ax.scatter([op["cov_disease"]],[op["acc_disease"]],color="#2E5496",zorder=5,s=40)
    ax.annotate("conformal α=0.2\n(guarded)",(op["cov_guarded"],op["acc_guarded"]),
                textcoords="offset points",xytext=(6,-24),fontsize=7.5,color="#B0413E")
    ax.set_xlabel("Coverage (fraction of subjects predicted)"); ax.set_ylabel("Selective accuracy")
    ax.set_xlim(0,1); ax.set_ylim(0.4,1.0); ax.legend(fontsize=8,frameon=False,loc="upper right")
    ax.set_title("Figure 5. Guarded ensemble with conformal abstention\n"
                 "The confound guard abstains on acquisition-explainable cases, lowering coverage;\n"
                 "retained accuracy stays near chance, showing little confound-free signal",fontsize=8.6)
    plt.tight_layout(); plt.savefig(OUTFIG,dpi=300,bbox_inches="tight"); plt.close()
    print("Figür:",OUTFIG)
    print(f"\nÖZET için: base_acc={base_acc:.3f}, guard passes {guard_mask.mean()*100:.0f}%,"
          f" guarded α=0.2 cov={op['cov_guarded']:.2f} acc={op['acc_guarded']:.3f}")

if __name__=="__main__":
    main()
