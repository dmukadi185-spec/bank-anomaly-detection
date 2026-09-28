"""
Bank Transaction Anomaly Detection
==================================
Pipeline de détection d'anomalies bancaires — version script (production).

IMPORTANT : ceci est de la DÉTECTION D'ANOMALIES, pas de la détection de fraude.
Le dataset ne contient pas de label de vérité (« fraude » ou non). On détecte donc
des transactions INHABITUELLES, qui sont des candidates à investigation humaine.

Méthode (honnête, sans overfitting) :
  1. Tri chronologique par compte (pas de fuite du futur)
  2. Ratio montant / solde AVANT la transaction
  3. Baseline historique par compte (moyenne passée)
  4. Modified Z-score sur log(ratio)  (robuste à l'asymétrie)
  5. Isolation Forest (anomalie multivariée)
  6. Score de risque pondéré, seuils auto-calibrés (quantile 0.98)
  7. Évaluation honnête : expliquabilité + ventilation (pas de faux 99%)

Usage :
    python src/anomaly_detection.py --input data/bank_transactions_data_2.csv

Sorties :
    output/investigation_prioritaire.csv   (top anomalies à investiguer)
    output/analyse_globale.csv             (toutes les transactions annotées)
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def load_and_prepare(path):
    """Charge, convertit les dates et trie chronologiquement par compte."""
    df = pd.read_csv(path)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df = df.sort_values(["AccountID", "TransactionDate"]).reset_index(drop=True)
    return df


def engineer_features(df):
    """Crée les features : solde avant, ratio, délai, baseline historique."""
    # Solde AVANT la transaction (AccountBalance est le solde POST-transaction)
    balance_avant = np.where(
        df["TransactionType"] == "Debit",
        df["AccountBalance"] + df["TransactionAmount"],
        df["AccountBalance"] - df["TransactionAmount"],
    )
    df["BalanceAvant"] = np.where(balance_avant > 0, balance_avant, df["AccountBalance"])
    df["CorrectRatio"] = df["TransactionAmount"] / df["BalanceAvant"].replace(0, np.nan)

    # Délai depuis la transaction précédente du même compte
    df["TimeDeltaMinutes"] = (
        df.groupby("AccountID")["TransactionDate"].diff().dt.total_seconds() / 60.0
    )

    # Baseline historique (moyenne des montants passés, sans fuite du futur)
    def past_mean(s):
        return s.shift(1).expanding(min_periods=3).mean()

    df["HistoricalMeanAmount"] = (
        df.groupby("AccountID")["TransactionAmount"].transform(past_mean)
    )
    df["HistoricalMeanAmount"] = df["HistoricalMeanAmount"].fillna(
        df["TransactionAmount"].mean()
    )
    df["AmountVsHistoryRatio"] = df["TransactionAmount"] / df["HistoricalMeanAmount"]

    return df


def detect_anomalies(df):
    """Applique Modified Z-score (log-ratio) + Isolation Forest + règles métier."""
    # Modified Z-score sur log(ratio) — robuste à l'asymétrie
    log_ratio = np.log1p(df["CorrectRatio"].fillna(0))
    median_log = log_ratio.median()
    mad_log = (log_ratio - median_log).abs().median() or 1e-6
    df["ModifiedZScore"] = 0.6745 * (log_ratio - median_log) / mad_log

    # Isolation Forest (multivarié)
    features = ["TransactionAmount", "CorrectRatio", "TimeDeltaMinutes", "LoginAttempts"]
    X = StandardScaler().fit_transform(df[features].fillna(0))
    iso = IsolationForest(contamination=0.03, random_state=42)
    df["IsoFlag"] = (iso.fit_predict(X) == -1).astype(int)

    # Règles avec seuils documentés + auto-calibrés
    z_threshold = df["ModifiedZScore"].abs().quantile(0.98)  # top 2 % extrême
    df["Rule_ZScore"] = (df["ModifiedZScore"].abs() > z_threshold).astype(int)
    df["Rule_Ratio"] = (df["CorrectRatio"] > 10.0).astype(int)
    df["Rule_Iso"] = df["IsoFlag"]

    # Score de risque pondéré
    df["RiskScore"] = 2 * df["Rule_ZScore"] + 3 * df["Rule_Ratio"] + 2 * df["Rule_Iso"]
    return df


def explain(row):
    """Explique POURQUOI une transaction est signalée (expliquabilité)."""
    reasons = []
    if row["Rule_Ratio"]:
        reasons.append("Ratio > 10x solde")
    if row["Rule_ZScore"]:
        reasons.append("Modified Z-Score extrême")
    if row["Rule_Iso"]:
        reasons.append("Anomalie multivariée (IsoForest)")
    return " | ".join(reasons) if reasons else "Normal"


def main(input_path):
    df = load_and_prepare(input_path)
    df = engineer_features(df)
    df = detect_anomalies(df)
    df["AnomalyReason"] = df.apply(explain, axis=1)

    anomalies = df[df["RiskScore"] > 0].sort_values("RiskScore", ascending=False)
    pct = len(anomalies) / len(df) * 100

    print("=" * 60)
    print("RAPPORT D'ÉVALUATION HONNÊTE")
    print("=" * 60)
    print(f"Volume analysé        : {len(df)} transactions")
    print(f"Anomalies signalées   : {len(anomalies)} ({pct:.1f} %)")
    print(f"  - Ratio > 10x       : {df['Rule_Ratio'].sum()}")
    print(f"  - Z-score extrême   : {df['Rule_ZScore'].sum()}")
    print(f"  - Isolation Forest  : {df['Rule_Iso'].sum()}")
    print()
    print(f"Conclusion : ~{pct:.1f} % signalés comme candidats à investigation.")
    print("(Anomalies = candidates, PAS des fraudes prouvées — pas de label de vérité)")

    # Exports
    cols = [
        "TransactionID", "AccountID", "TransactionDate", "TransactionType",
        "TransactionAmount", "BalanceAvant", "CorrectRatio", "RiskScore", "AnomalyReason",
    ]
    anomalies[cols].to_csv("output/investigation_prioritaire.csv", index=False)
    df.to_csv("output/analyse_globale.csv", index=False)
    print()
    print("[OK] output/investigation_prioritaire.csv")
    print("[OK] output/analyse_globale.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection d'anomalies bancaires")
    parser.add_argument("--input", default="data/bank_transactions_data_2.csv")
    args = parser.parse_args()
    main(args.input)
