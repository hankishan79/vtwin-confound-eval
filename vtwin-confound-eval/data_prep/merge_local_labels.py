# -*- coding: utf-8 -*-
r"""
merge_local_labels.py
retrain_extract.js'in ürettiği KANONİK voice45_v1 session CSV'sine (filename + 45 feature
+ QC) Birlesik_all.xlsx'ten site/etiket/yaş bilgisini ekler ve analize hazır CSV yazar.

Akış:
  1) PowerShell'de:  npm install
                     node scripts/retrain_extract.js --input "D:\Datasetler\DATASETLER\WAVs" --output features_local_voice45.csv
  2) python merge_local_labels.py   (bu script)
  -> features_local_voice45_labeled.csv  (ibni_cross_cohort.py / ibni_within_cohort_audit.py ile uyumlu)

Not: Çıkarım Node ile yapıldığı için feature'lar web ile BİT-AYNI (train-serve skew yok).
"""
import os, numpy as np, pandas as pd

NODE_CSV = r"features_local_voice45.csv"            # retrain_extract.js çıktısı (session-level)
XLSX     = r"D:\Datasetler\DATASETLER\Birlesik_all.xlsx"
SHEET    = "Birlesik_Dataset_Sablonu"
OUT_CSV  = "features_local_voice45_labeled.csv"

FEAT = ([f"mfcc_{i}" for i in range(13)] + [f"delta_mfcc_{i}" for i in range(13)] +
        ["spectral_centroid_mean","spectral_flatness_mean","spectral_rolloff_mean",
         "spectral_flux_mean","rms_mean","zcr_mean","energy_mean"] +
        ["spectral_centroid_std","spectral_flatness_std","spectral_rolloff_std",
         "rms_std","zcr_std"] +
        ["f0_mean","f0_std","jitter","shimmer","hnr","voiced_fraction","breathiness"])
QC = ["duration_sec","overall_dbfs","speech_dbfs","estimated_snr_db",
      "speech_frame_fraction","selected_segment_count"]


def build_label_map(xlsx):
    df = pd.read_excel(xlsx, sheet_name=SHEET).iloc[1:].reset_index(drop=True)
    df["LC"]=pd.to_numeric(df["LUNG_CANCER"],errors="coerce")
    df["CO"]=pd.to_numeric(df["COPD"],errors="coerce")
    def lab(r):
        if r["LC"]==1: return "LC"
        if r["CO"]==1: return "COPD"
        if r["LC"]==0: return "NON_LC"
        return "UNLABELED"
    df["label"]=df.apply(lab,axis=1)
    def site_norm(s):
        s=str(s).strip().upper()
        if s.startswith("İBNİ") or s.startswith("IBNI") or "BNI" in s.replace("İ","I"): return "IBNI"
        return s
    df["site_inferred"]=df["VERİ_KAYNAĞI"].apply(site_norm)
    df["_voice_id"]=df["VOICE_ID"].astype(str).str.strip()
    df["_data_id"]=df["DATA_ID"].astype(str).str.strip()
    df["_subj"]=df["SUBJ_NO"].astype(str).str.strip()
    try: df["_duz"]=pd.to_numeric(df["DÜZENLENMİŞ VOICE_ID"],errors="coerce").astype("Int64").astype(str)
    except Exception: df["_duz"]=df["DÜZENLENMİŞ VOICE_ID"].astype(str)
    return df


def match_row(stem, df):
    for col in ["_voice_id","_data_id","_duz","_subj"]:
        hit = df[df[col]==stem]
        if len(hit)==1: return hit.iloc[0]
    for col in ["_voice_id","_data_id"]:
        vals=[str(v) for v in df[col].tolist()]
        idxs=[i for i,v in enumerate(vals) if v not in ("nan","") and v in stem]
        if len(idxs)==1: return df.iloc[idxs[0]]
    return None


def main():
    feats = pd.read_csv(NODE_CSV)
    miss=[c for c in FEAT if c not in feats.columns]
    if miss:
        raise ValueError(f"{NODE_CSV} 45 feature kolonu eksik (ilk 5): {miss[:5]} — doğru session CSV mi?")
    lab = build_label_map(XLSX)

    rows=[]; unmatched=[]
    for _, r in feats.iterrows():
        fn = str(r["filename"]); stem=os.path.splitext(os.path.basename(fn))[0]
        m = match_row(stem, lab)
        rec = dict(filename=fn)
        for c in FEAT+[q for q in QC if q in feats.columns]:
            rec[c]=r[c]
        # Node QC kolon adları aynı (duration_sec, overall_dbfs, ...) → varsa taşı
        if m is not None:
            rec.update(subj_no=m["_subj"], site=str(m["VERİ_KAYNAĞI"]).strip(),
                       site_inferred=m["site_inferred"], label=m["label"],
                       AGE=pd.to_numeric(m["AGE"],errors="coerce"), GENDER=m["GENDER"],
                       SMOKING=pd.to_numeric(m["SMOKING"],errors="coerce"),
                       DM=pd.to_numeric(m["DM"],errors="coerce"), HT=pd.to_numeric(m["HT"],errors="coerce"))
        else:
            unmatched.append(fn)
            rec.update(subj_no=None, site=None, site_inferred="UNMATCHED", label="UNMATCHED",
                       AGE=np.nan, GENDER=None, SMOKING=np.nan, DM=np.nan, HT=np.nan)
        rows.append(rec)

    out=pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Yazıldı → {OUT_CSV}  (n={len(out)})  | eşleşmeyen={len(unmatched)}")
    if unmatched[:8]: print("  örnek eşleşmeyen dosya:", unmatched[:8])
    print("\nlabel × site_inferred:")
    print(pd.crosstab(out["site_inferred"], out["label"]))


if __name__ == "__main__":
    main()
