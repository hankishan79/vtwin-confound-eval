# -*- coding: utf-8 -*-
"""
paired_contrast_transfer.py — V-TWIN — MEŞRU ANALİZ #1
=====================================================================
ÖN-KAYIT (sonuçlara bakmadan ÖNCE yazıldı):

  Hipotez: Cross-cohort transferin şans/şans-altı olmasının bir nedeni, "non-LC"
  kontrast tanımının iki kohortta FARKLI olması olabilir:
     - A (Local321): non-LC = karışık (healthy + COPD + other)
     - B (Web2026):  non-LC = komorbid hasta kontrolleri (sağlıklı DEĞİL)
  Yani "LC-vs-nonLC" iki kohortta farklı bir görev. Kontrastı TUTARLI hale getirip
  (her ikisinde de LC-vs-komorbid) transferin düzelip düzelmediğine bakacağız.

  Karar kuralı (ÖN-TANIMLI, çalıştırmadan önce sabit):
    * Tutarlı kontrast altında PHYS-transfer her iki yönde >= 0.60 ve
      source-leakage düşerse → kontrast-uyumsuzluğu pay sahibiydi, temkinli pozitif.
    * Transfer hâlâ ~0.50/şans-altı kalırsa → kontrast-uyumsuzluğu açıklamıyor,
      Claim C çelikleşir.
  HER İKİ SONUÇ DA RAPORLANIR. Kontrast tanımı bir kez seçilir; "düzelene kadar"
  alt-grup denemesi YAPILMAZ (= HARKing).

Çıktı: konsol tablosu + results/paired_contrast_results.json/.csv
"""

import json, os
import numpy as np
import pandas as pd
import da_common as dc

# ==========================================================================
# CONFIG — kendi kolon adlarınıza göre doldurun
# ==========================================================================
# 45 öznitelik kolonu, LAYOUT SIRASINDA. Aşağıdaki adlar TAHMİNDİR; kendi
# features_local.csv / web tablosu başlığınızla birebir eşleştirin.
FEATURE_COLS = (
    [f"mfcc_{i}" for i in range(13)] +
    [f"dmfcc_{i}" for i in range(13)] +
    ["sc_mean", "sb_mean", "sro_mean", "sf_mean", "rms_mean", "zcr_mean", "energy_mean"] +
    ["sc_std", "sb_std", "rms_std", "zcr_std", "energy_std"] +
    ["f0_mean", "f0_std", "jitter", "shimmer", "hnr", "voiced_fraction", "breathiness"]
)
assert len(FEATURE_COLS) == 45

COHORT_A = dict(
    name="Local321",
    path=r"D:\Makale Calismalari\NEW\features_local.csv",
    feature_cols=FEATURE_COLS,
    label_col="label",            # <-- LC/non-LC etiket kolonu
    label_lc_value="LC",          # <-- LC'yi gösteren değer
    subgroup_col="subgroup",      # <-- healthy/copd/other/... ; YOKSA None yapın
    age_col=None,                 # eşleştirme istemiyorsanız None
    sex_col=None,
)

COHORT_B = dict(
    name="Web2026",
    path=r"website_subjects_parsed.csv",
    feature_cols=FEATURE_COLS,
    label_col="label",
    label_lc_value="LC",
    subgroup_col=None,            # B'de non-LC zaten komorbid → harmonizasyon gerekmez
    age_col=None,
    sex_col=None,
)

# Hangi alt-gruplar "komorbid kontrol" sayılacak (tutarlı kontrast).
# A'da healthy ATILIR; sadece komorbid hastalar kontrol olur (B ile uyumlu).
COMORBID_CONTROL_SUBGROUPS = {"copd", "other", "comorbid", "asthma", "bronchitis"}
HEALTHY_SUBGROUPS = {"healthy", "control", "normal", "saglikli", "sağlıklı"}

# Demografik eşleştirme (ops.). True ise kohort-içinde kontrolleri vakalara
# (sex, age-bin) ile coarsened-exact eşler. age_col/sex_col gerektirir.
DO_DEMOGRAPHIC_MATCHING = False
AGE_BIN_EDGES = [0, 50, 60, 70, 200]

RESULTS_DIR = "results"


# ==========================================================================
def harmonize_to_comorbid(coh):
    """A için: healthy at, sadece LC + komorbid bırak. subgroup yoksa olduğu gibi döner."""
    if coh["sub"] is None:
        return coh
    sub = coh["sub"]
    is_lc = coh["y"] == 1
    is_comorbid_ctrl = (coh["y"] == 0) & np.isin(sub, list(COMORBID_CONTROL_SUBGROUPS))
    keep = is_lc | is_comorbid_ctrl
    dropped_healthy = int(np.sum((coh["y"] == 0) & np.isin(sub, list(HEALTHY_SUBGROUPS))))
    dropped_unknown = int(np.sum((coh["y"] == 0) & ~np.isin(
        sub, list(COMORBID_CONTROL_SUBGROUPS | HEALTHY_SUBGROUPS))))
    out = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in coh.items()}
    out["_dropped_healthy"] = dropped_healthy
    out["_dropped_unknown_ctrl"] = dropped_unknown
    return out


def demographic_match(coh, seed=dc.RANDOM_SEED):
    """Kohort-içi coarsened-exact eşleştirme: her (sex, age-bin) hücresinde
    kontrol sayısını vaka sayısına indir. age/sex yoksa değiştirmeden döner."""
    if coh["age"] is None or coh["sex"] is None:
        return coh
    rng = np.random.default_rng(seed)
    age_bin = np.digitize(coh["age"], AGE_BIN_EDGES)
    strata = np.array([f"{s}|{b}" for s, b in zip(coh["sex"], age_bin)])
    keep_idx = []
    for st in np.unique(strata):
        in_st = np.where(strata == st)[0]
        cases = in_st[coh["y"][in_st] == 1]
        ctrls = in_st[coh["y"][in_st] == 0]
        if len(cases) == 0 or len(ctrls) == 0:
            continue  # eşleşmeyen hücreyi at
        k = min(len(ctrls), max(len(cases), 1))
        sel_ctrls = rng.choice(ctrls, size=k, replace=False)
        keep_idx.extend(cases.tolist()); keep_idx.extend(sel_ctrls.tolist())
    keep_idx = np.sort(np.array(keep_idx, int))
    return {k: (v[keep_idx] if isinstance(v, np.ndarray) else v) for k, v in coh.items()}


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    A = dc.load_cohort(**{k: v for k, v in COHORT_A.items() if k != "name"})
    B = dc.load_cohort(**{k: v for k, v in COHORT_B.items() if k != "name"})

    print("=" * 70)
    print("MEŞRU ANALİZ #1 — Eşleştirilmiş-kontrast transfer (ÖN-KAYITLI)")
    print("=" * 70)
    print(f"Ham:  A({COHORT_A['name']}) n={len(A['y'])} LC={A['y'].sum()} | "
          f"B({COHORT_B['name']}) n={len(B['y'])} LC={B['y'].sum()}")

    A = harmonize_to_comorbid(A)
    B = harmonize_to_comorbid(B)
    if "_dropped_healthy" in A:
        print(f"Harmonizasyon A: healthy atıldı={A['_dropped_healthy']}, "
              f"bilinmeyen-kontrol atıldı={A['_dropped_unknown_ctrl']}")

    if DO_DEMOGRAPHIC_MATCHING:
        A = demographic_match(A); B = demographic_match(B)
        print("Demografik eşleştirme uygulandı (coarsened-exact: sex × age-bin).")

    print(f"Harmonize: A n={len(A['y'])} LC={A['y'].sum()} ctrl={(A['y']==0).sum()} | "
          f"B n={len(B['y'])} LC={B['y'].sum()} ctrl={(B['y']==0).sum()}")
    print("-" * 70)

    rows = []
    for sname, idx in dc.SUBSETS.items():
        Xa, Xb = dc._sub(A["X"], idx), dc._sub(B["X"], idx)
        cv_a = dc.internal_cv_auc(Xa, A["y"])
        cv_b = dc.internal_cv_auc(Xb, B["y"])
        t_ab = dc.transfer_auc(Xa, A["y"], Xb, B["y"])
        t_ba = dc.transfer_auc(Xb, B["y"], Xa, A["y"])
        leak = dc.source_leakage_auc(Xa, Xb)
        rows.append(dict(subset=sname, cv_A=cv_a, cv_B=cv_b,
                         transfer_A2B=t_ab, transfer_B2A=t_ba, source_leakage=leak))
        print(f"[{sname:6}] CV_A={cv_a:.3f} CV_B={cv_b:.3f} | "
              f"A->B={t_ab:.3f} B->A={t_ba:.3f} | leakage={leak:.3f}")

    print("-" * 70)
    phys = next(r for r in rows if r["subset"] == "phys")
    verdict_pos = (phys["transfer_A2B"] >= 0.60 and phys["transfer_B2A"] >= 0.60)
    print("KARAR (ön-tanımlı): "
          + ("TEMKİNLİ POZİTİF — tutarlı kontrast altında PHYS transfer >=0.60 "
             "(kontrast-uyumsuzluğu pay sahibiydi). Leakage'e de bakın."
             if verdict_pos else
             "Claim C ÇELİKLEŞİR — tutarlı kontrast altında bile PHYS transfer şans civarı; "
             "kontrast-uyumsuzluğu açıklamıyor, edinim-baskınlığı sürüyor."))

    out = dict(config=dict(harmonized_to="LC-vs-comorbid",
                           demographic_matching=DO_DEMOGRAPHIC_MATCHING),
               n=dict(A=int(len(A["y"])), B=int(len(B["y"]))),
               results=rows, predefined_verdict_positive=bool(verdict_pos))
    with open(os.path.join(RESULTS_DIR, "paired_contrast_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "paired_contrast_results.csv"), index=False)
    print(f"\nKaydedildi → {RESULTS_DIR}/paired_contrast_results.[json|csv]")


if __name__ == "__main__":
    run()
