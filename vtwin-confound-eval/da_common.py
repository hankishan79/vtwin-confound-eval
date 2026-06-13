# -*- coding: utf-8 -*-
"""
da_common.py — V-TWIN
Ortak yardımcılar: kohort yükleme, SABİT model, internal-CV / transfer / source-leakage
primitifleri, CORAL hizalama ve öznitelik alt-kümeleri.

İLKE (her iki analiz için):
  * Model SABİT ve ÖN-TANIMLI: StandardScaler + LogisticRegression(C=1.0).
    Transfer iyileşene kadar C/model/öznitelik AYARLAMA. Bir kez, dürüst raporla.
  * Hedef (target) ETİKETLERİNE eğitim sırasında ASLA dokunulmaz.
  * Öznitelik alt-kümeleri 45-D Voice45 layout'una göre POZİSYONLA tanımlı
    (kolon adından bağımsız), cross_cohort.py ile birebir aynı olmalı.

45-D layout:
  [0-12]  MFCC
  [13-25] delta-MFCC
  [26-32] spectral/rms/zcr/energy (mean)
  [33-37] std'ler
  [38-39] F0 mean/std
  [40] jitter [41] shimmer [42] HNR [43] voiced_fraction [44] breathiness
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------
# Öznitelik alt-kümeleri (0-tabanlı indeksler, layout'a göre)
# cross_cohort.py'deki tanımlarla AYNI olmalı; farklıysa burada düzeltin.
# --------------------------------------------------------------------------
IDX_FULL   = list(range(45))
IDX_PHYS   = [38, 39, 40, 41, 42, 43, 44]                 # f0m,f0s,jitter,shimmer,hnr,voiced,breath
IDX_DEVICE = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]  # spektral/rms/zcr/energy + std (cihaz imzası)

SUBSETS = {"full": IDX_FULL, "phys": IDX_PHYS, "device": IDX_DEVICE}

RANDOM_SEED = 0


# --------------------------------------------------------------------------
# Yükleme
# --------------------------------------------------------------------------
def load_cohort(path, feature_cols, label_col, label_lc_value,
                subgroup_col=None, age_col=None, sex_col=None, sep=","):
    """
    Bir kohortu yükler ve standart bir sözlük döndürür.

    feature_cols : 45 öznitelik kolon adı (sıralı, layout ile aynı sıra!)
    label_col    : etiket kolonu
    label_lc_value: bu değer LC(=1) sayılır; diğer her şey non-LC(=0)
    subgroup_col : (ops.) 'healthy'/'copd'/'other'/'comorbid' gibi alt-grup; eşleştirilmiş-kontrast için
    age_col,sex_col: (ops.) demografik eşleştirme için
    """
    df = pd.read_csv(path, sep=sep)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: eksik öznitelik kolonları (ilk 10): {missing[:10]}")
    if label_col not in df.columns:
        raise ValueError(f"{path}: etiket kolonu yok: {label_col}")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = (df[label_col].astype(str).str.strip() == str(label_lc_value)).astype(int).to_numpy()

    sub = df[subgroup_col].astype(str).str.strip().str.lower().to_numpy() if subgroup_col else None
    age = pd.to_numeric(df[age_col], errors="coerce").to_numpy() if age_col else None
    sex = df[sex_col].astype(str).str.strip().str.lower().to_numpy() if sex_col else None

    # NaN öznitelik satırlarını at (kolon-bazlı medyan yerine satır-at: dürüstlük; istenirse değiştir)
    keep = ~np.isnan(X).any(axis=1)
    out = dict(X=X[keep], y=y[keep],
               sub=(sub[keep] if sub is not None else None),
               age=(age[keep] if age is not None else None),
               sex=(sex[keep] if sex is not None else None))
    return out


# --------------------------------------------------------------------------
# Sabit model
# --------------------------------------------------------------------------
def make_model():
    """ÖN-TANIMLI sabit model. C'yi/modeli değiştirmeyin."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])


def _sub(X, idx):
    return X[:, idx]


# --------------------------------------------------------------------------
# Metrik primitifleri
# --------------------------------------------------------------------------
def internal_cv_auc(X, y, n_splits=5, seed=RANDOM_SEED):
    """Kohort-içi stratified k-fold AUC (out-of-fold)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(make_model(), X, y, cv=skf, method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def transfer_auc(Xs, ys, Xt, yt):
    """Source'ta eğit, target'ta test (target etiketi sadece skorlama için kullanılır)."""
    if len(np.unique(ys)) < 2 or len(np.unique(yt)) < 2:
        return float("nan")
    m = make_model().fit(Xs, ys)
    proba = m.predict_proba(Xt)[:, 1]
    return roc_auc_score(yt, proba)


def source_leakage_auc(Xa, Xb, n_splits=5, seed=RANDOM_SEED):
    """Sesten kohort tahmini (A=0,B=1) için CV AUC. 0.5'e yakın = sızıntı yok."""
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa), int), np.ones(len(Xb), int)]
    return internal_cv_auc(X, y, n_splits=n_splits, seed=seed)


# --------------------------------------------------------------------------
# CORAL (unsupervised, target etiketi YOK)
# --------------------------------------------------------------------------
def _sym_power(C, p):
    """Simetrik PSD matris için C^p (eigh ile sayısal olarak kararlı)."""
    vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
    vals = np.clip(vals, 1e-10, None)
    return (vecs * (vals ** p)) @ vecs.T


def coral_align(Xs, Xt, eps=1.0):
    """
    Source'u target'ın 2. derece istatistiklerine hizala (CORAL).
    eps ÖN-TANIMLI sabit (regularizasyon). Çalıştırdıktan sonra DEĞİŞTİRMEYİN.
    """
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(d)
    Xs_white = Xs @ _sym_power(Cs, -0.5)
    Xs_aligned = Xs_white @ _sym_power(Ct, 0.5)
    return np.real(Xs_aligned)


def transfer_auc_coral(Xs, ys, Xt, yt, eps=1.0):
    """
    CORAL'lı transfer. Akış:
      1) StandardScaler source'ta fit → source & target standardize
      2) source'u target'a CORAL-hizala (target ETİKETİ kullanılmaz)
      3) LogisticRegression hizalı source'ta eğit, standardize target'ta test
    Ayrıca hizalama-sonrası source-leakage'i de döndürür.
    """
    if len(np.unique(ys)) < 2 or len(np.unique(yt)) < 2:
        return float("nan"), float("nan")
    sc = StandardScaler().fit(Xs)
    Xs_s, Xt_s = sc.transform(Xs), sc.transform(Xt)
    Xs_a = coral_align(Xs_s, Xt_s, eps=eps)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(Xs_a, ys)
    auc = roc_auc_score(yt, clf.predict_proba(Xt_s)[:, 1])

    # hizalama-sonrası sızıntı: hizalı source vs standardize target ayrılabiliyor mu?
    leak = source_leakage_auc(Xs_a, Xt_s)
    return auc, leak
