# -*- coding: utf-8 -*-
"""
ibni_cross_cohort.py — V-TWIN — IBNI(A) ↔ Web2026(B) cross-cohort
================================================================================
FİKİR (Dr. Ankışhan): full Local321 yerine SADECE IBNI'yi kohort A yap; böylece
Acibadem/Ankara merkez-sinyali A'dan çıkar ve cross-cohort daha temiz olur.

ÖN-KAYIT (sonuca bakmadan): IBNI-only A ile transfer, full-Local321 A'ya göre
DEĞİŞİR mi? Karşılaştırma için full-Local321 versiyonunu da çalıştırın (cross_cohort.py).

UYARI (dürüstlük): IBNI-içi ana confound MERKEZLER-ARASI değil; YAŞ (kontroller ~18 yaş
daha genç) ve KAYIT-SÜRESİ/ses-seviyesi (COPD koleksiyonu). Bu yüzden ham transferin
yanında residualized(QC+AGE) transfer de raporlanır. NET-bio transfer asıl ölçüttür.

Çalıştırma sırası önerisi:
  1) ibni_within_cohort_audit.py  (IBNI-içi confound tablosu — referans)
  2) bu script                    (IBNI ↔ Web transfer)
Çıktı: results/ibni_cross_cohort.csv + konsol
"""
import os, json, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
import da_common as dc

# ---- gerçek voice45_v1 kolon adları (web ve IBNI dosyaları AYNI şemada) ----
FEAT = ([f"mfcc_{i}" for i in range(13)] + [f"delta_mfcc_{i}" for i in range(13)] +
        ["spectral_centroid_mean", "spectral_flatness_mean", "spectral_rolloff_mean",
         "spectral_flux_mean", "rms_mean", "zcr_mean", "energy_mean"] +
        ["spectral_centroid_std", "spectral_flatness_std", "spectral_rolloff_std",
         "rms_std", "zcr_std"] +
        ["f0_mean", "f0_std", "jitter", "shimmer", "hnr", "voiced_fraction", "breathiness"])
assert len(FEAT) == 45
CONF_CANDIDATES = ["duration_sec", "overall_dbfs", "speech_dbfs",
                   "speech_frame_fraction", "selected_segment_count", "AGE"]

# ==========================================================================
# CONFIG
# ==========================================================================
# A = IBNI (full Local321 dosyasından site_inferred=='IBNI' filtrelenir)
IBNI_FILE   = r"features_local_voice45_labeled.csv"   # yeni kanonik (Node voice45_v1)
IBNI_SITE_COL = "site_inferred"; IBNI_SITE_VALUE = "IBNI"
IBNI_LABEL_COL = "label"; IBNI_LC_VALUE = "LC"
# Kontrast: B (web) non-LC = komorbid hasta. A tarafında non-LC'yi şöyle tanımla:
#   "NON_LC_only"  -> sadece NON_LC (COPD hariç)
#   "NON_LC_COPD"  -> NON_LC + COPD (tüm non-LC)
A_NEG_DEF = "NON_LC_COPD"

# B = Web2026 (subject-level). ChatGPT'nin ürettiği split-label dosyasını verin.
WEB_FILE  = r"website_subjects_parsed_splitlabel.csv"   # 100 subject, label LC/nonLC
WEB_LABEL_COL = "label"; WEB_LC_VALUE = "LC"             # nonLC otomatik non-LC(=0) olur

RESULTS_DIR = "results"


def load_features(path, label_col, lc_value, site_col=None, site_value=None,
                  neg_labels=None):
    df = pd.read_csv(path)
    if site_col and site_col in df.columns and site_value is not None:
        df = df[df[site_col] == site_value].copy()
    miss = [c for c in FEAT if c not in df.columns]
    if miss:
        raise ValueError(f"{path}: eksik öznitelik kolonları (ilk 8): {miss[:8]}")
    Xf = df[FEAT].apply(pd.to_numeric, errors="coerce")
    keep = ~Xf.isna().any(axis=1)
    df, Xf = df[keep], Xf[keep]
    lab = df[label_col].astype(str).str.strip()
    is_lc = (lab == str(lc_value)).values
    if neg_labels is not None:  # A için: sadece LC + belirtilen neg etiketleri tut
        keep2 = is_lc | lab.isin(neg_labels).values
        df, Xf, is_lc = df[keep2], Xf[keep2], is_lc[keep2]
    y = is_lc.astype(int)
    conf = [c for c in CONF_CANDIDATES if c in df.columns]
    C = df[conf].apply(pd.to_numeric, errors="coerce") if conf else None
    return Xf.to_numpy(float), y, C, conf


def residualize_transfer(Xs, ys, Xt, yt, Cs, Ct):
    """Source'ta confound->öznitelik fit; source & target residüalize; transfer.
    AYRICA residualizasyon-SONRASI source-leakage döndürür:
      leak hâlâ yüksek (>~0.7) ise → transfer confound'u yeniden kodluyor (ARTEFAKT),
      leak düştü ise (→0.5) → kaymayla gizlenmiş sinyal açığa çıkmış olabilir.
    Sadece HER İKİ kohortta ortak confound varsa anlamlı."""
    if Cs is None or Ct is None or Cs.shape[1] == 0:
        return np.nan, np.nan
    Cs = Cs.copy(); Ct = Ct.copy()
    for C in (Cs, Ct):
        C.fillna(C.median(), inplace=True)
    lr = LinearRegression().fit(Cs.values, Xs)
    Xs_r = Xs - lr.predict(Cs.values)
    Xt_r = Xt - lr.predict(Ct.values)
    auc = dc.transfer_auc(Xs_r, ys, Xt_r, yt)
    leak_resid = dc.source_leakage_auc(Xs_r, Xt_r)   # residual uzayda kohort hâlâ ayrılıyor mu?
    return auc, leak_resid


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    neg = ["NON_LC"] if A_NEG_DEF == "NON_LC_only" else ["NON_LC", "COPD"]
    Xa, ya, Ca, ca = load_features(IBNI_FILE, IBNI_LABEL_COL, IBNI_LC_VALUE,
                                   IBNI_SITE_COL, IBNI_SITE_VALUE, neg_labels=neg)
    Xb, yb, Cb, cb = load_features(WEB_FILE, WEB_LABEL_COL, WEB_LC_VALUE)

    common_conf = [c for c in ca if c in cb]
    Ca_c = Ca[common_conf] if common_conf else None
    Cb_c = Cb[common_conf] if common_conf else None

    print("=" * 74)
    print("IBNI(A) ↔ Web2026(B) cross-cohort transfer")
    print("=" * 74)
    print(f"A=IBNI  n={len(ya)} LC={ya.sum()} neg={int((ya==0).sum())} ({A_NEG_DEF})")
    print(f"B=Web   n={len(yb)} LC={yb.sum()} neg={int((yb==0).sum())}")
    print(f"Ortak confound (residualize için): {common_conf or 'YOK'}")
    print("-" * 74)
    print(f"{'subset':7}{'CV_A':>8}{'CV_B':>8}{'A->B':>8}{'B->A':>8}"
          f"{'A->B_res':>10}{'B->A_res':>10}{'leak':>8}")

    rows = []
    for sname, idx in dc.SUBSETS.items():
        Xa_s, Xb_s = dc._sub(Xa, idx), dc._sub(Xb, idx)
        cva = dc.internal_cv_auc(Xa_s, ya)
        cvb = dc.internal_cv_auc(Xb_s, yb)
        t_ab = dc.transfer_auc(Xa_s, ya, Xb_s, yb)
        t_ba = dc.transfer_auc(Xb_s, yb, Xa_s, ya)
        # residualized sadece full öznitelikte anlamlı (confound tüm uzayı etkiler)
        if sname == "full" and common_conf:
            r_ab, rleak_ab = residualize_transfer(Xa, ya, Xb, yb, Ca_c, Cb_c)
            r_ba, rleak_ba = residualize_transfer(Xb, yb, Xa, ya, Cb_c, Ca_c)
        else:
            r_ab = r_ba = rleak_ab = rleak_ba = np.nan
        leak = dc.source_leakage_auc(Xa_s, Xb_s)
        rows.append(dict(subset=sname, CV_A=cva, CV_B=cvb, transfer_A2B=t_ab,
                         transfer_B2A=t_ba, transfer_A2B_resid=r_ab,
                         transfer_B2A_resid=r_ba, leak_resid_A2B=rleak_ab,
                         leak_resid_B2A=rleak_ba, source_leakage=leak))
        print(f"{sname:7}{cva:>8.3f}{cvb:>8.3f}{t_ab:>8.3f}{t_ba:>8.3f}"
              f"{r_ab:>10.3f}{r_ba:>10.3f}{leak:>8.3f}")
    # residualizasyon-sonrası leakage (artefakt vs gerçek sinyal testi)
    full_row = next(r for r in rows if r["subset"] == "full")
    if common_conf:
        print(f"\n[diagnostik] residualize edilen confound: {common_conf}")
        print(f"[diagnostik] residualizasyon-SONRASI leakage: "
              f"A->B={full_row['leak_resid_A2B']:.3f}  B->A={full_row['leak_resid_B2A']:.3f}")
        print("  yüksek (>~0.7) → residualized transfer confound'u YENİDEN KODLUYOR (artefakt);"
              " 0.5'e yakın → gizlenmiş sinyal açığa çıkmış olabilir.")

    print("-" * 74)
    phys = next(r for r in rows if r["subset"] == "phys")
    print("YORUM rehberi (ön-tanımlı):")
    print("  * PHYS transfer her iki yön >=0.60 ve full-Local321'e göre belirgin "
          "ARTTI ise → merkez-confound'u pay sahibiydi; IBNI-only meşru iyileştirme.")
    print("  * Hâlâ ~0.50 ise → confound merkez değil (yaş/süre/koleksiyon); "
          "IBNI-only transferi kurtarmıyor, Claim C sürüyor.")
    print(f"  (Bu çalıştırmada PHYS A->B={phys['transfer_A2B']:.3f}, "
          f"B->A={phys['transfer_B2A']:.3f}, leakage(full)="
          f"{next(r for r in rows if r['subset']=='full')['source_leakage']:.3f})")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ibni_cross_cohort.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "ibni_cross_cohort.json"), "w") as f:
        json.dump(dict(A_neg=A_NEG_DEF, common_conf=common_conf, results=rows), f, indent=2)
    print(f"\nKaydedildi → {RESULTS_DIR}/ibni_cross_cohort.[csv|json]")


if __name__ == "__main__":
    main()
