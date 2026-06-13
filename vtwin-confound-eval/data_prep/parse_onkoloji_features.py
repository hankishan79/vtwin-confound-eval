# -*- coding: utf-8 -*-
"""
parse_onkoloji_features.py — Onkoloji_ses.csv'deki HAZIR voice45_v1 feature'larını
(Node/Meyda ile zaten çıkarılmış) ve QC/klinik alanları schema-uyumlu CSV'ye çevirir.
WAV çıkarımı GEREKMEZ. Bu feature'lar web kohortuyla AYNI çıkarıcıdan gelir.

⚠️ Etiket otoritesi: clinical_data.lung_cancer alanı GÜVENİLMEZ (web-formu, çoğu false).
Bu yüzden 'lung_cancer_form' ham olarak yazılır; gerçek LC etiketi manuel kürasyondan
gelmeli. Hepsi onkoloji-merkezi kaydı olduğundan 'label_assumed=LC' eklenir (doğrula).

Çalıştır: python parse_onkoloji_features.py
"""
import json, ast, os
import numpy as np, pandas as pd

IN_CSV  = r"Onkoloji_ses.csv"
OUT_CSV = "features_onkoloji_voice45.csv"

FEAT = ([f"mfcc_{i}" for i in range(13)] + [f"delta_mfcc_{i}" for i in range(13)] +
        ["spectral_centroid_mean","spectral_flatness_mean","spectral_rolloff_mean",
         "spectral_flux_mean","rms_mean","zcr_mean","energy_mean"] +
        ["spectral_centroid_std","spectral_flatness_std","spectral_rolloff_std",
         "rms_std","zcr_std"] +
        ["f0_mean","f0_std","jitter","shimmer","hnr","voiced_fraction","breathiness"])


def _loadjson(x):
    if not isinstance(x, str) or not x.strip():
        return {}
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}


def parse_vector(x):
    if not isinstance(x, str):
        return None
    try:
        v = json.loads(x)
    except Exception:
        try:
            v = ast.literal_eval(x)
        except Exception:
            return None
    v = list(v)
    return v if len(v) == 45 else None


def main():
    df = pd.read_csv(IN_CSV)
    rows = []
    skipped = 0
    for _, r in df.iterrows():
        vec = parse_vector(r.get("feature_vector"))
        if vec is None:
            skipped += 1
            continue
        rec = {f: float(vec[i]) for i, f in enumerate(FEAT)}
        qc = _loadjson(r.get("quality_report")).get("metrics", {})
        cl = _loadjson(r.get("clinical_data"))
        ctx = _loadjson(r.get("recording_context"))
        rec.update(
            subject_code=r.get("subject_code"),
            site="WEB_ONKOL",
            label_assumed="LC",                       # onkoloji merkezi → doğrula
            lung_cancer_form=cl.get("lung_cancer"),    # GÜVENİLMEZ ham alan
            copd_form=cl.get("copd"),
            AGE=pd.to_numeric(cl.get("age"), errors="coerce"),
            GENDER=cl.get("gender"),
            SMOKING=cl.get("smoking"),
            duration_sec=qc.get("durationSec"),
            overall_dbfs=qc.get("overallDbfs"),
            speech_dbfs=qc.get("speechDbfs"),
            estimated_snr_db=qc.get("estimatedSnrDb"),
            speech_frame_fraction=qc.get("speechFrameFraction"),
            selected_segment_count=qc.get("speechFrameCount"),
            device_id=ctx.get("deviceId"),
            recording_plan=ctx.get("recordingPlan"),
            feature_spec=r.get("feature_spec_version"),
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Yazıldı → {OUT_CSV}  (n={len(out)}, atlanan={skipped})")
    print("device_id dağılımı:\n", out["device_id"].value_counts(dropna=False))
    print("\nlung_cancer_form (ham/güvenilmez):", out["lung_cancer_form"].value_counts(dropna=False).to_dict())
    print("Yaş özet:", out["AGE"].describe()[["mean","min","max","count"]].round(1).to_dict())


if __name__ == "__main__":
    main()
