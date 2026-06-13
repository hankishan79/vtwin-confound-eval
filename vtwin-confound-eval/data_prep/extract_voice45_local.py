# -*- coding: utf-8 -*-
"""
extract_voice45_local.py — V-TWIN V-TWIN pipeline
Yerel WAV'lardan 45-boyutlu Voice45 + QC öznitelikleri çıkarır, Birlesik_all.xlsx'ten
etiket/site/yaş bilgisini eşler ve downstream scriptlerle uyumlu bir CSV yazar.

⚠️ TRAIN-SERVE SKEW UYARISI
Bu Python çıkarıcı, web tarafının Node/Meyda 'voice45_v1' çıktısıyla SAYISAL OLARAK
AYNI DEĞİLDİR (ör. parselmouth HNR ~dB iken voice45_v1 HNR ~1-3 ölçeğinde). Bu yüzden:
  • Yerel WAV'lar ARASI (merkez-içi / İbn-i Sina-içi) analiz için GEÇERLİ (hepsi aynı çıkarıcı).
  • Yerel ↔ web cross-cohort için DOĞRUDAN KARŞILAŞTIRMAYIN. O karşılaştırma için ya
    (a) web WAV'larını da bu scriptle yeniden çıkarın, ya da (b) yerel WAV'ları Node
    'retrain_extract.js' ile çıkarın (önerilen, çünkü web zaten voice45_v1).

Kurulum:  pip install librosa praat-parselmouth soundfile pandas openpyxl numpy
Çalıştır: python extract_voice45_local.py
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import librosa
try:
    import parselmouth
    from parselmouth.praat import call
    HAVE_PM = True
except Exception:
    HAVE_PM = False

# ==========================================================================
# CONFIG
# ==========================================================================
WAV_DIR   = r"D:\Datasetler\DATASETLER\WAVs"
XLSX      = r"D:\Datasetler\DATASETLER\Birlesik_all.xlsx"
SHEET     = "Birlesik_Dataset_Sablonu"
OUT_CSV   = "features_local_voice45_py.csv"
SR        = 16000          # tüm dosyalar bu örnekleme hızına getirilir (tutarlılık)
PREEMPH   = 0.97
FRAME_MS, HOP_MS = 25, 10
N_MFCC    = 13

FEAT = ([f"mfcc_{i}" for i in range(13)] + [f"delta_mfcc_{i}" for i in range(13)] +
        ["spectral_centroid_mean","spectral_flatness_mean","spectral_rolloff_mean",
         "spectral_flux_mean","rms_mean","zcr_mean","energy_mean"] +
        ["spectral_centroid_std","spectral_flatness_std","spectral_rolloff_std",
         "rms_std","zcr_std"] +
        ["f0_mean","f0_std","jitter","shimmer","hnr","voiced_fraction","breathiness"])

# ==========================================================================
def extract_features(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")
    # peak-norm + pre-emphasis (orijinal pipeline ile aynı sıra)
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    y_pe = np.append(y[0], y[1:] - PREEMPH * y[:-1])

    n_fft = int(SR * FRAME_MS / 1000); hop = int(SR * HOP_MS / 1000)
    S = np.abs(librosa.stft(y_pe, n_fft=n_fft, hop_length=hop)) + 1e-10

    mfcc = librosa.feature.mfcc(y=y_pe, sr=SR, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop)
    dmfcc = librosa.feature.delta(mfcc)
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR)[0]
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0)); flux = np.append(flux, flux[-1] if flux.size else 0)
    rms  = librosa.feature.rms(S=S, frame_length=n_fft)[0]
    zcr  = librosa.feature.zero_crossing_rate(y_pe, frame_length=n_fft, hop_length=hop)[0]
    energy = (librosa.util.frame(y_pe, frame_length=n_fft, hop_length=hop) ** 2).sum(axis=0)

    f = {}
    for i in range(13): f[f"mfcc_{i}"] = float(mfcc[i].mean())
    for i in range(13): f[f"delta_mfcc_{i}"] = float(dmfcc[i].mean())
    f["spectral_centroid_mean"]=float(cent.mean()); f["spectral_flatness_mean"]=float(flat.mean())
    f["spectral_rolloff_mean"]=float(roll.mean());  f["spectral_flux_mean"]=float(flux.mean())
    f["rms_mean"]=float(rms.mean()); f["zcr_mean"]=float(zcr.mean()); f["energy_mean"]=float(energy.mean())
    f["spectral_centroid_std"]=float(cent.std()); f["spectral_flatness_std"]=float(flat.std())
    f["spectral_rolloff_std"]=float(roll.std()); f["rms_std"]=float(rms.std()); f["zcr_std"]=float(zcr.std())

    # --- phonation (parselmouth/praat) ---
    voiced_frac = np.nan; f0m=f0s=jit=shim=hnr=brth=np.nan
    if HAVE_PM:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
        pitch = snd.to_pitch(time_step=HOP_MS/1000.0, pitch_floor=75, pitch_ceiling=500)
        f0 = pitch.selected_array['frequency']; vmask = f0 > 0
        voiced_frac = float(vmask.mean()) if f0.size else np.nan
        if vmask.sum() > 1:
            f0m=float(f0[vmask].mean()); f0s=float(f0[vmask].std())
        try:
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jit = float(call(pp, "Get jitter (local)", 0,0, 1e-4, 0.02, 1.3))
            shim= float(call([snd, pp], "Get shimmer (local)", 0,0, 1e-4, 0.02, 1.3, 1.6))
        except Exception: pass
        try:
            harm = snd.to_harmonicity_cc(time_step=HOP_MS/1000.0, minimum_pitch=75)
            hv = harm.values[harm.values != -200]
            hnr = float(hv.mean()) if hv.size else np.nan
        except Exception: pass
    else:  # parselmouth yoksa librosa.pyin fallback (jitter/shimmer/hnr kaba)
        f0, vflag, _ = librosa.pyin(y_pe, fmin=75, fmax=500, sr=SR, hop_length=hop)
        vmask = ~np.isnan(f0); voiced_frac=float(np.nanmean(vflag)) if vflag.size else np.nan
        if vmask.sum()>1: f0m=float(np.nanmean(f0[vmask])); f0s=float(np.nanstd(f0[vmask]))
    # breathiness proxy = sesli çerçevelerde ortalama spektral düzlük (yüksek=daha gürültülü/nefesli)
    brth = float(flat.mean())
    f["f0_mean"]=f0m; f["f0_std"]=f0s; f["jitter"]=jit; f["shimmer"]=shim
    f["hnr"]=hnr; f["voiced_fraction"]=voiced_frac; f["breathiness"]=brth

    # --- QC / acquisition descriptors (çerçeve-RMS genliği üzerinden, tutarlı dBFS) ---
    dur = len(y) / SR
    overall_dbfs = 20*np.log10(np.sqrt(np.mean(y**2)) + 1e-12)
    frame_rms = rms                                    # STFT çerçeve genliği (0..~1)
    thr = np.percentile(frame_rms, 40)                 # basit enerji-tabanlı VAD eşiği
    speech = frame_rms > thr
    sp_rms = frame_rms[speech]; ns_rms = frame_rms[~speech]
    speech_dbfs = 20*np.log10(sp_rms.mean() + 1e-12) if sp_rms.size else np.nan
    snr = 20*np.log10((sp_rms.mean() + 1e-12)/(ns_rms.mean() + 1e-12)) if (sp_rms.size and ns_rms.size) else np.nan
    seg_count = (int(np.sum(np.diff(speech.astype(int)) == 1)) + int(speech[0])) if speech.size else 0
    f["duration_sec"]=float(dur); f["overall_dbfs"]=float(overall_dbfs)
    f["speech_dbfs"]=float(speech_dbfs); f["speech_frame_fraction"]=float(speech.mean())
    f["estimated_snr_db"]=float(snr); f["selected_segment_count"]=int(seg_count)
    return f


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
    # eşleştirme için aday ID'ler (string)
    df["_voice_id"]=df["VOICE_ID"].astype(str).str.strip()
    df["_data_id"]=df["DATA_ID"].astype(str).str.strip()
    df["_subj"]=df["SUBJ_NO"].astype(str).str.strip()
    try: df["_duz"]=pd.to_numeric(df["DÜZENLENMİŞ VOICE_ID"],errors="coerce").astype("Int64").astype(str)
    except Exception: df["_duz"]=df["DÜZENLENMİŞ VOICE_ID"].astype(str)
    return df


def match_wav_to_row(stem, df):
    """WAV dosya kökünü Excel satırıyla eşle (sırayla: voice_id, data_id, duz, subj)."""
    for col in ["_voice_id","_data_id","_duz","_subj"]:
        hit = df[df[col]==stem]
        if len(hit)==1: return hit.iloc[0]
    # substring denemesi (voice_id / data_id dosya adının içinde mi)
    for col in ["_voice_id","_data_id"]:
        vals = [str(v) for v in df[col].tolist()]
        idxs = [i for i, v in enumerate(vals) if v not in ("nan","") and v in stem]
        if len(idxs) == 1:
            return df.iloc[idxs[0]]
    return None


def main():
    df = build_label_map(XLSX)
    wavs = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")) +
                  glob.glob(os.path.join(WAV_DIR, "*.WAV")))
    print(f"{len(wavs)} WAV bulundu. parselmouth: {'VAR' if HAVE_PM else 'YOK (fallback)'}")
    rows, unmatched, failed = [], [], []
    for k, w in enumerate(wavs, 1):
        stem = os.path.splitext(os.path.basename(w))[0]
        r = match_wav_to_row(stem, df)
        try:
            feats = extract_features(w)
        except Exception as e:
            failed.append((os.path.basename(w), str(e))); continue
        meta = dict(filename=os.path.basename(w), stem=stem)
        if r is not None:
            meta.update(subj_no=r["_subj"], site=str(r["VERİ_KAYNAĞI"]).strip(),
                        label=r["label"], AGE=pd.to_numeric(r["AGE"],errors="coerce"),
                        GENDER=r["GENDER"], SMOKING=pd.to_numeric(r["SMOKING"],errors="coerce"),
                        DM=pd.to_numeric(r["DM"],errors="coerce"), HT=pd.to_numeric(r["HT"],errors="coerce"))
        else:
            unmatched.append(os.path.basename(w))
            meta.update(subj_no=None, site=None, label="UNMATCHED", AGE=np.nan,
                        GENDER=None, SMOKING=np.nan, DM=np.nan, HT=np.nan)
        meta.update(feats); rows.append(meta)
        if k % 25 == 0: print(f"  ... {k}/{len(wavs)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nYazıldı → {OUT_CSV}  (n={len(out)})")
    print("Eşleşmeyen WAV:", len(unmatched), "| Çıkarımı başarısız:", len(failed))
    if unmatched[:5]: print("  örnek eşleşmeyen:", unmatched[:5])
    if failed[:5]:    print("  örnek başarısız:", failed[:5])
    if "label" in out: print("\nEtiket dağılımı:\n", out["label"].value_counts())
    if "site" in out:  print("\nSite dağılımı:\n", out["site"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
