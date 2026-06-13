# -*- coding: utf-8 -*-
"""
find_web_file.py — klasördeki CSV'ler arasından doğru WEB_FILE'ı bulur.
Çalıştır: python find_web_file.py     (ibni_cross_cohort.py ile aynı klasörde)
45 voice45 kolonunu ve olası etiket kolonunu içeren dosyaları listeler.
"""
import os, glob, pandas as pd

FEAT = ([f"mfcc_{i}" for i in range(13)] + [f"delta_mfcc_{i}" for i in range(13)] +
        ["spectral_centroid_mean", "spectral_flatness_mean", "spectral_rolloff_mean",
         "spectral_flux_mean", "rms_mean", "zcr_mean", "energy_mean"] +
        ["spectral_centroid_std", "spectral_flatness_std", "spectral_rolloff_std",
         "rms_std", "zcr_std"] +
        ["f0_mean", "f0_std", "jitter", "shimmer", "hnr", "voiced_fraction", "breathiness"])
LABEL_CANDIDATES = ["label", "label_meta", "split_label", "class", "group", "target", "y"]

print(f"Klasör: {os.getcwd()}\n")
csvs = sorted(glob.glob("*.csv"))
if not csvs:
    print("Bu klasörde .csv yok. ibni_cross_cohort.py ile aynı klasörde çalıştırın.")
hits = []
for f in csvs:
    try:
        head = pd.read_csv(f, nrows=200)
    except Exception as e:
        print(f"[okunamadı] {f}: {e}"); continue
    n_feat = sum(c in head.columns for c in FEAT)
    lab_cols = [c for c in LABEL_CANDIDATES if c in head.columns]
    if n_feat >= 40:  # voice45 dosyası gibi
        nrows = sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) - 1
        info = f"  >>> {f}  | voice45 kolonu={n_feat}/45 | satır≈{nrows}"
        if lab_cols:
            lc = lab_cols[0]
            full_lab = pd.read_csv(f, usecols=[lc])[lc]
            vc = full_lab.astype(str).str.strip().value_counts().to_dict()
            info += f" | etiket='{lc}' örnek değerler={vc}"
        else:
            info += " | UYARI: etiket kolonu bulunamadı"
        hits.append(info)

print("=== voice45 öznitelikli CSV adayları ===")
if hits:
    print("\n".join(hits))
    print("\nWEB_FILE'ı yukarıdaki '>>>' satırlarından LC/non-LC içeren dosyaya ayarlayın.")
    print("Etiket kolonu 'label' değilse, WEB_LABEL_COL ve WEB_LC_VALUE'yu da güncelleyin.")
else:
    print("voice45 kolonlarına sahip CSV bulunamadı. Web öznitelik dosyası başka klasörde olabilir;")
    print("bu klasöre kopyalayın ya da WEB_FILE'a tam yolu yazın (ör. r'D:\\...\\dosya.csv').")
