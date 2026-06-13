# -*- coding: utf-8 -*-
"""
coral_transfer.py — V-TWIN — MEŞRU ANALİZ #2
===========================================================
ÖN-KAYIT (sonuçlara bakmadan ÖNCE yazıldı):

  Hipotez: Modern domain-invariant hizalama (CORAL) cross-cohort transferi meşru
  biçimde artırabilir mi? CORAL source'un 2. derece istatistiklerini target'a
  hizalar; target ETİKETİ KULLANILMAZ (unsupervised DA → dürüst robustness testi).

  Karar kuralı (ÖN-TANIMLI, eps=1.0 sabit, TEK çalıştırma):
    * CORAL sonrası transfer hâlâ ~0.50 ise → "modern domain-invariant yöntemler
      bile başarısız" → Claim C ÇELİKLEŞİR.
    * CORAL transferi belirgin artırır (>=0.60) VE hizalama-sonrası source-leakage
      düşerse → kaymayla GİZLENMİŞ gerçek sinyal vardı → TEMKİNLİ POZİTİF;
      doğrulama: within-platform + ön-tanımlı tekrar gerekir.
    * Transfer artar AMA leakage hâlâ yüksekse → muhtemelen confound'u yeniden
      kodluyor, sinyal değil → güvenme.
  TEK ÇALIŞTIRMA. eps'i/akışı "düzelene kadar" oynatmak HARKing'tir; sonuç ne
  olursa olsun raporlanır.

NOT: CORAL "olmayan sinyali yaratamaz", sadece kaymayla gizlenmişi açığa çıkarır.
Source-leakage 0.94 + şans-altı PHYS transfer (ham) zaten düşük tavan öngörüyor.

Çıktı: konsol tablosu + results/coral_results.json/.csv
"""

import json, os
import numpy as np
import pandas as pd
import da_common as dc

# CONFIG'i paired_contrast ile aynı tutmak için oradan içe alıyoruz.
from paired_contrast_transfer import (
    FEATURE_COLS, COHORT_A, COHORT_B, RESULTS_DIR,
)

CORAL_EPS = 1.0  # ÖN-TANIMLI, SABİT. Çalıştırdıktan sonra DEĞİŞTİRMEYİN.


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # CORAL, manşet transfer GÖREVİYLE (orijinal LC-vs-nonLC) çalıştırılır;
    # harmonizasyon UYGULANMAZ (#1 ile karşılaştırılabilir kalsın diye).
    A = dc.load_cohort(**{k: v for k, v in COHORT_A.items() if k != "name"})
    B = dc.load_cohort(**{k: v for k, v in COHORT_B.items() if k != "name"})

    print("=" * 74)
    print("MEŞRU ANALİZ #2 — Tek-sefer CORAL domain-adaptation (ÖN-KAYITLI, eps=1.0)")
    print("=" * 74)
    print(f"Görev: orijinal LC-vs-nonLC. A n={len(A['y'])} LC={A['y'].sum()} | "
          f"B n={len(B['y'])} LC={B['y'].sum()}")
    print("-" * 74)
    print(f"{'subset':7}{'A->B raw':>10}{'A->B coral':>12}{'B->A raw':>10}"
          f"{'B->A coral':>12}{'leak raw':>10}{'leak A2B':>10}{'leak B2A':>10}")

    rows = []
    for sname, idx in dc.SUBSETS.items():
        Xa, Xb = dc._sub(A["X"], idx), dc._sub(B["X"], idx)
        raw_ab = dc.transfer_auc(Xa, A["y"], Xb, B["y"])
        raw_ba = dc.transfer_auc(Xb, B["y"], Xa, A["y"])
        leak_raw = dc.source_leakage_auc(Xa, Xb)

        cor_ab, leak_ab = dc.transfer_auc_coral(Xa, A["y"], Xb, B["y"], eps=CORAL_EPS)
        cor_ba, leak_ba = dc.transfer_auc_coral(Xb, B["y"], Xa, A["y"], eps=CORAL_EPS)

        rows.append(dict(subset=sname,
                         A2B_raw=raw_ab, A2B_coral=cor_ab,
                         B2A_raw=raw_ba, B2A_coral=cor_ba,
                         leak_raw=leak_raw, leak_coral_A2B=leak_ab, leak_coral_B2A=leak_ba))
        print(f"{sname:7}{raw_ab:>10.3f}{cor_ab:>12.3f}{raw_ba:>10.3f}"
              f"{cor_ba:>12.3f}{leak_raw:>10.3f}{leak_ab:>10.3f}{leak_ba:>10.3f}")

    print("-" * 74)
    phys = next(r for r in rows if r["subset"] == "phys")
    improved = (phys["A2B_coral"] >= 0.60 and phys["B2A_coral"] >= 0.60)
    leak_dropped = (phys["leak_coral_A2B"] < 0.70 and phys["leak_coral_B2A"] < 0.70)
    if improved and leak_dropped:
        verdict = ("TEMKİNLİ POZİTİF — CORAL PHYS transferini >=0.60'a çıkardı VE "
                   "leakage düştü; kaymayla gizlenmiş sinyal olabilir. Within-platform + "
                   "ön-tanımlı tekrarla DOĞRULA.")
    elif improved and not leak_dropped:
        verdict = ("ŞÜPHELİ — transfer arttı ama leakage hâlâ yüksek; muhtemelen "
                   "confound'u yeniden kodluyor. GÜVENME.")
    else:
        verdict = ("Claim C ÇELİKLEŞİR — modern domain-invariant hizalama bile PHYS "
                   "transferi şans civarında bırakıyor.")
    print("KARAR (ön-tanımlı):", verdict)

    out = dict(config=dict(coral_eps=CORAL_EPS, task="LC-vs-nonLC (orijinal)"),
               n=dict(A=int(len(A["y"])), B=int(len(B["y"]))),
               results=rows, predefined_verdict=verdict)
    with open(os.path.join(RESULTS_DIR, "coral_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "coral_results.csv"), index=False)
    print(f"\nKaydedildi → {RESULTS_DIR}/coral_results.[json|csv]")


if __name__ == "__main__":
    run()
