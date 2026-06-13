# -*- coding: utf-8 -*-
"""
ibni_within_cohort_audit.py — V-TWIN
IBNI-içi (tek-merkez) confound-kontrollü ana biyolojik analiz.

AMAÇ: "IBNI'yi tek kohort yap" fikrini dürüstçe test etmek. IBNI, full Local321'e
göre Acibadem/Ankara merkez-sinyalini ÇIKARIR — ama IBNI-içinde kendi confound'ları
vardır (LC/COPD/kontrol farklı KOLEKSİYONLARDAN; yaş ve kayıt-süresi etiketle karışık).
Bu script her kontrast için Voice45 vs QC-only vs AGE-only vs residualized(QC+AGE)
karşılaştırır ve sesin alt-koleksiyonu ne kadar ele verdiğini ölçer.

Girdi : features_local_ibni_candidate_subjectlevel.csv  (veya site_inferred=='IBNI')
Çıktı : results/ibni_within_cohort_audit.csv  + konsol
"""
import os, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

RS = 0
IBNI_PATH = "features_local_voice45_labeled.csv"   # yeni kanonik (Node voice45_v1) + etiket
COLLECTION_COL = "site"          # alt-koleksiyon kimliği (İBNİ SİNA / İBNİ SİNA 2 / İBNİ SİNA 3)
RESULTS_DIR = "results"

FEAT = ([f"mfcc_{i}" for i in range(13)] + [f"delta_mfcc_{i}" for i in range(13)] +
        ["spectral_centroid_mean", "spectral_flatness_mean", "spectral_rolloff_mean",
         "spectral_flux_mean", "rms_mean", "zcr_mean", "energy_mean"] +
        ["spectral_centroid_std", "spectral_flatness_std", "spectral_rolloff_std",
         "rms_std", "zcr_std"] +
        ["f0_mean", "f0_std", "jitter", "shimmer", "hnr", "voiced_fraction", "breathiness"])
assert len(FEAT) == 45
PHYS, DEVICE = FEAT[38:45], FEAT[26:38]
QC = ["duration_sec", "overall_dbfs", "speech_dbfs", "speech_frame_fraction", "selected_segment_count"]
CONF = QC + ["AGE"]   # residualizasyonda çıkarılacak confound'lar


def model():
    return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=1.0))])


def cv_auc(X, y, seed=RS, k=5):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    k = min(k, int(np.min(np.bincount(y))))
    if k < 2:
        return np.nan
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    p = cross_val_predict(model(), X, y, cv=skf, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def cv_auc_residualized(Xdf, y, conf_df, seed=RS, k=5):
    """Fold-içi residualizasyon: confound->öznitelik lineer fit SADECE train'de.
    Sızıntısız NET-bio tahmini."""
    y = np.asarray(y)
    X = Xdf.to_numpy(float).copy()
    C = conf_df.to_numpy(float).copy()
    med = np.nanmedian(C, axis=0)
    nanpos = np.where(np.isnan(C))
    C[nanpos] = np.take(med, nanpos[1])
    k = min(k, int(np.min(np.bincount(y))))
    if k < 2:
        return np.nan
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        lr = LinearRegression().fit(C[tr], X[tr])
        Xr_tr = X[tr] - lr.predict(C[tr])
        Xr_te = X[te] - lr.predict(C[te])
        m = model().fit(Xr_tr, y[tr])
        oof[te] = m.predict_proba(Xr_te)[:, 1]
    return roc_auc_score(y, oof)


def contrast(df, pos, neg, name, rows):
    d = df[df["label"].isin(pos + neg)].copy()
    Xf = d[FEAT].apply(pd.to_numeric, errors="coerce")
    keep = ~Xf.isna().any(axis=1)
    d, Xf = d[keep], Xf[keep]
    y = d["label"].isin(pos).astype(int).values
    r = dict(contrast=name, n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
             voice45_full=cv_auc(Xf.values, y),
             voice45_phys=cv_auc(d[PHYS].values, y),
             voice45_device=cv_auc(d[DEVICE].values, y),
             qc_only=cv_auc(d[QC].fillna(d[QC].median()).values, y),
             age_only=cv_auc(d[["AGE"]].fillna(d["AGE"].median()).values, y),
             net_bio_resid_QC_AGE=cv_auc_residualized(Xf, y, d[CONF]))
    rows.append(r)
    print(f"\n### {name}  (pos n={r['n_pos']} | neg n={r['n_neg']})")
    print(f"  Voice45 full={r['voice45_full']:.3f}  phys={r['voice45_phys']:.3f}  "
          f"device={r['voice45_device']:.3f}")
    print(f"  QC-only={r['qc_only']:.3f}  AGE-only={r['age_only']:.3f}  "
          f"[negatif kontroller]")
    print(f"  NET-bio resid(QC+AGE)={r['net_bio_resid_QC_AGE']:.3f}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(IBNI_PATH)
    if "site_inferred" in df.columns and (df["site_inferred"] == "IBNI").any():
        df = df[df["site_inferred"] == "IBNI"].copy()
    print("=" * 72)
    print("IBNI-içi confound-kontrollü analiz (subject-level)")
    print("=" * 72)
    print("Etiket dağılımı:", df["label"].value_counts().to_dict())

    rows = []
    contrast(df, ["LC"], ["NON_LC"], "LC vs NON_LC", rows)
    contrast(df, ["LC"], ["COPD"], "LC vs COPD", rows)
    contrast(df, ["LC"], ["NON_LC", "COPD"], "LC vs all non-LC/COPD", rows)

    # Koleksiyon sızıntısı
    print("\n" + "=" * 72)
    print("IBNI-içi KOLEKSİYON sızıntısı (sesten/QC'den alt-koleksiyon tahmini)")
    print("=" * 72)
    leak_rows = []
    Xf = df[FEAT].apply(pd.to_numeric, errors="coerce")
    keep = ~Xf.isna().any(axis=1)
    dd, Xf = df[keep], Xf[keep]
    for grp in dd[COLLECTION_COL].unique():
        yy = (dd[COLLECTION_COL] == grp).astype(int).values
        if yy.sum() < 5:
            continue
        a_v = cv_auc(Xf.values, yy)
        a_q = cv_auc(dd[QC].fillna(dd[QC].median()).values, yy)
        leak_rows.append(dict(group=grp, n=int(yy.sum()), voice45_auc=a_v, qc_auc=a_q))
        print(f"  {grp:30s} (n={yy.sum():3d}) voice45={a_v:.3f}  QC={a_q:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ibni_within_cohort_audit.csv"), index=False)
    pd.DataFrame(leak_rows).to_csv(os.path.join(RESULTS_DIR, "ibni_collection_leakage.csv"), index=False)
    print(f"\nKaydedildi → {RESULTS_DIR}/ibni_within_cohort_audit.csv + ibni_collection_leakage.csv")


if __name__ == "__main__":
    main()
