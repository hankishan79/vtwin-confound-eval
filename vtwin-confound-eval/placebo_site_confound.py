# -*- coding: utf-8 -*-
"""
placebo_site_confound.py — V-TWIN confound-aware framework
Çok-merkezli "placebo" gösterimi: ses-temelli ayrımın hastalık değil, MERKEZ/YAŞ
kaynaklı olduğunu somutlar. Üç analiz:

  (1) Site-leakage: sesten/QC'den merkez tahmini AUC (yüksek = edinim imzası).
  (2) Sham (kanser-YOK) merkez-arası kontrast: iki kanser-içermeyen grubu (IBNI
      kontrolleri vs başka-merkez gönüllüleri) ayırma AUC'si. Hastalık farkı YOK;
      0.5'ten yüksek her şey saf merkez/yaş artefaktı.
  (3) LC tespitinde DÜRÜST vs NAİF kontrol: IBNI LC'yi (a) kendi IBNI kontrolleriyle,
      (b) başka-merkez gönüllüleriyle ayırma; naif kontrolün AUC'yi nasıl şişirdiğini
      ve QC/AGE ile açıklandığını gösterir.

Girdi : features_local_voice45_labeled.csv  (Node voice45_v1 + etiket/site)
Çıktı : results/placebo_site_confound_*.csv + konsol
"""
import os, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

RS = 0
PATH = "features_local_voice45_labeled.csv"
RESULTS_DIR = "results"
NAIVE_CONTROL_SITE = "ACIBADEM"     # naif/çapraz-merkez kontrol kaynağı (genç gönüllüler)
MIN_SITE_N = 20                     # site-leakage için minimum n

FEAT = ([f"mfcc_{i}" for i in range(13)] + [f"delta_mfcc_{i}" for i in range(13)] +
        ["spectral_centroid_mean","spectral_flatness_mean","spectral_rolloff_mean",
         "spectral_flux_mean","rms_mean","zcr_mean","energy_mean"] +
        ["spectral_centroid_std","spectral_flatness_std","spectral_rolloff_std",
         "rms_std","zcr_std"] +
        ["f0_mean","f0_std","jitter","shimmer","hnr","voiced_fraction","breathiness"])
PHYS, DEVICE = FEAT[38:45], FEAT[26:38]
QC = ["duration_sec","overall_dbfs","speech_dbfs","speech_frame_fraction","selected_segment_count"]


def model():
    return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=1.0))])


def auc(X, y, k=5):
    y = np.asarray(y)
    if len(np.unique(y)) < 2: return np.nan
    k = min(k, int(np.min(np.bincount(y))))
    if k < 2: return np.nan
    p = cross_val_predict(model(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=RS),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def clean(df):
    Xf = df[FEAT].apply(pd.to_numeric, errors="coerce")
    keep = ~Xf.isna().any(axis=1)
    return df[keep].copy()


def panel(d, y, name, rows):
    """full/phys/device/QC-only/AGE-only AUC paneli."""
    r = dict(contrast=name, n_pos=int(np.sum(y == 1)), n_neg=int(np.sum(y == 0)),
             full=auc(d[FEAT].values, y), phys=auc(d[PHYS].values, y),
             device=auc(d[DEVICE].values, y),
             qc_only=auc(d[QC].fillna(d[QC].median()).values, y),
             age_only=auc(d[["AGE"]].fillna(d["AGE"].median()).values, y))
    rows.append(r)
    print(f"  [{name:30s}] n=({r['n_pos']},{r['n_neg']}) full={r['full']:.3f} "
          f"phys={r['phys']:.3f} device={r['device']:.3f} QC={r['qc_only']:.3f} AGE={r['age_only']:.3f}")
    return r


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = clean(pd.read_csv(PATH))
    df["site_inferred"] = df["site_inferred"].astype(str)
    print("Site dağılımı:", df["site_inferred"].value_counts().to_dict())

    # ---------- (1) SITE-LEAKAGE ----------
    print("\n(1) SITE-LEAKAGE — sesten/QC'den merkez tahmini (yüksek = edinim imzası)")
    leak_rows = []
    for s in df["site_inferred"].value_counts().index:
        sub = df[df["site_inferred"] == s]
        if len(sub) < MIN_SITE_N: continue
        y = (df["site_inferred"] == s).astype(int).values
        a_v = auc(df[FEAT].values, y); a_q = auc(df[QC].fillna(df[QC].median()).values, y)
        leak_rows.append(dict(site=s, n=int(len(sub)), voice45_auc=a_v, qc_auc=a_q))
        print(f"  {s:26s} (n={len(sub):3d}) voice45={a_v:.3f}  QC={a_q:.3f}")
    pd.DataFrame(leak_rows).to_csv(os.path.join(RESULTS_DIR, "placebo_site_leakage.csv"), index=False)

    # ---------- (2) SHAM (kanser-YOK) merkez-arası kontrast ----------
    print("\n(2) SHAM kontrast — iki KANSER-İÇERMEYEN grup (hastalık farkı YOK)")
    ibni_ctrl = df[(df["site_inferred"] == "IBNI") & (df["label"] == "NON_LC")]
    other_ctrl = df[df["site_inferred"] == NAIVE_CONTROL_SITE]   # etiketsiz gönüllüler = kanser yok
    sham = pd.concat([ibni_ctrl, other_ctrl], ignore_index=True)
    y_sham = np.r_[np.zeros(len(ibni_ctrl), int), np.ones(len(other_ctrl), int)]
    sham_rows = []
    if len(ibni_ctrl) >= 10 and len(other_ctrl) >= 10:
        print(f"  IBNI-kontrol (yaş ort {ibni_ctrl['AGE'].mean():.1f}) vs "
              f"{NAIVE_CONTROL_SITE} (yaş ort {other_ctrl['AGE'].mean():.1f})")
        panel(sham, y_sham, f"SHAM IBNIctrl_vs_{NAIVE_CONTROL_SITE}", sham_rows)
        print("  → Hastalık farkı yok; full/phys'in 0.5'ten yüksek olması SAF merkez/yaş artefaktı.")
    else:
        print("  (yetersiz n — atlandı)")
    pd.DataFrame(sham_rows).to_csv(os.path.join(RESULTS_DIR, "placebo_sham_contrast.csv"), index=False)

    # ---------- (3) LC: DÜRÜST vs NAİF kontrol ----------
    print("\n(3) LC tespiti — DÜRÜST (IBNI-içi) vs NAİF (çapraz-merkez) kontrol")
    lc = df[(df["site_inferred"] == "IBNI") & (df["label"] == "LC")]
    rows3 = []
    # (a) dürüst: IBNI LC vs IBNI NON_LC
    d_h = pd.concat([lc, ibni_ctrl], ignore_index=True)
    y_h = np.r_[np.ones(len(lc), int), np.zeros(len(ibni_ctrl), int)]
    panel(d_h, y_h, "LC vs IBNI-control (HONEST)", rows3)
    # (b) naif: IBNI LC vs başka-merkez gönüllüler
    d_n = pd.concat([lc, other_ctrl], ignore_index=True)
    y_n = np.r_[np.ones(len(lc), int), np.zeros(len(other_ctrl), int)]
    panel(d_n, y_n, f"LC vs {NAIVE_CONTROL_SITE}-control (NAIVE)", rows3)
    print("  → Naif kontrolde full AUC şişer; QC-only/AGE-only onu açıklar (sinyal değil, confound).")
    pd.DataFrame(rows3).to_csv(os.path.join(RESULTS_DIR, "placebo_honest_vs_naive.csv"), index=False)

    print(f"\nKaydedildi → {RESULTS_DIR}/placebo_site_leakage.csv, placebo_sham_contrast.csv, "
          f"placebo_honest_vs_naive.csv")


if __name__ == "__main__":
    main()
