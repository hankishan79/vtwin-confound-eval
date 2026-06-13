# -*- coding: utf-8 -*-
"""_selftest.py — sentetik veriyle kodun hatasız çalıştığını doğrular (gerçek sonuç DEĞİL)."""
import numpy as np, pandas as pd, os, tempfile
import paired_contrast_transfer as pc
import da_common as dc

rng = np.random.default_rng(0)

def make_cohort(n, lc_frac, device_offset, signal, with_sub):
    y = (rng.random(n) < lc_frac).astype(int)
    X = rng.normal(0, 1, size=(n, 45))
    # zayıf LC sinyali phys indekslerinde
    for j in dc.IDX_PHYS:
        X[:, j] += signal * y
    # kohort-spesifik cihaz ofseti (source leakage simülasyonu)
    for j in dc.IDX_DEVICE:
        X[:, j] += device_offset
    df = pd.DataFrame(X, columns=pc.FEATURE_COLS)
    df["label"] = np.where(y == 1, "LC", "nonLC")
    if with_sub:
        sub = np.where(y == 1, "lc",
              np.where(rng.random(n) < 0.5, "healthy", "copd"))
        df["subgroup"] = sub
    return df

tmp = tempfile.mkdtemp()
pa = os.path.join(tmp, "A.csv"); pb = os.path.join(tmp, "B.csv")
make_cohort(300, 0.18, device_offset=+3.0, signal=0.8, with_sub=True).to_csv(pa, index=False)
# B'de sinyal TERS yönde (cross-cohort uyumsuzluğu simülasyonu) + farklı cihaz ofseti
make_cohort(110, 0.65, device_offset=-3.0, signal=-0.8, with_sub=False).to_csv(pb, index=False)

# CONFIG'i sentetik dosyalara yönlendir
pc.COHORT_A.update(path=pa, subgroup_col="subgroup")
pc.COHORT_B.update(path=pb, subgroup_col=None)
pc.RESULTS_DIR = os.path.join(tmp, "results")

print("\n########## ANALİZ #1 ##########")
pc.run()

# coral, import anında pc'den COHORT_A/B'yi aldı; aynı dict objesi → güncel
import coral_transfer as ct
ct.RESULTS_DIR = pc.RESULTS_DIR
print("\n########## ANALİZ #2 ##########")
ct.run()
print("\n[OK] Her iki script de hatasız çalıştı.")
