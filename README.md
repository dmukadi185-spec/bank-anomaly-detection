# 🏦 Bank Transaction Anomaly Detection

Détection d'**anomalies** dans les transactions bancaires à l'aide de Python (pandas, scikit-learn).

> ⚠️ **Clarification honnête** : ce projet détecte des **anomalies** (transactions inhabituelles), pas des **fraudes prouvées**. Le dataset ne contient aucun label de vérité (`fraud` / `not fraud`). Les anomalies sont donc des **candidates à investigation humaine**, pas des verdicts.

---

## 🎯 Objectif

Identifier les transactions statistiquement inhabituelles dans un dataset de 2 512 opérations bancaires, en combinant **règles métier**, **statistiques robustes** et **détection multivariée (Isolation Forest)**.

## 🧠 Méthodologie (7 étapes, sans overfitting)

| # | Étape | Technique |
|---|---|---|
| 1 | Tri chronologique par compte | évite la fuite du futur (lookahead) |
| 2 | Ratio `montant / solde AVANT` | solde post-transaction corrigé |
| 3 | Baseline historique par compte | moyenne passée (causalité correcte) |
| 4 | **Modified Z-score** sur `log(ratio)` | robuste à l'asymétrie (log-normal) |
| 5 | **Isolation Forest** | anomalie multivariée (montant, ratio, délai, logins) |
| 6 | Score de risque pondéré | seuils **documentés + auto-calibrés** (quantile 0.98) |
| 7 | **Évaluation honnête** | expliquabilité + ventilation (pas de faux 99 %) |

## 📊 Résultats

```
Volume analysé        : 2 512 transactions
Anomalies signalées   : 103 (4.1 %)

  - Ratio > 10x       : 3
  - Z-score extrême   : 51
  - Isolation Forest  : 76

Conclusion : ~4 % des transactions sont des candidates à investigation
```

Chaque anomalie est **expliquée** (`AnomalyReason`) : on sait *pourquoi* elle est signalée (ratio, z-score, ou anomalie multivariée).

## ⚠️ Limites assumées

1. **Pas de label de vérité** → impossible de calculer une accuracy. C'est de l'analyse **non supervisée**.
2. **Le ratio montant/solde est plus pertinent pour les Débits** (retraits) que les Crédits (dépôts).
3. Le dataset est **synthétique** (Kaggle) — les seuils sont calibrés pour la démonstration, pas pour la production.

## 🚀 Utilisation

### Notebook (exploratoire)
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Bank_visualisation.ipynb
```

### Script (production)
```bash
python src/anomaly_detection.py --input data/bank_transactions_data_2.csv
```

## 📁 Structure

```
bank-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Bank_visualisation.ipynb     # analyse exploratoire + pipeline complet
├── src/
│   └── anomaly_detection.py         # pipeline en script (production)
├── data/
│   └── README.md                    # source du dataset (non inclus)
└── output/                          # rapports générés (CSV)
```

## 📦 Données

Dataset Kaggle : [Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection) (2 512 transactions, 16 features). **Non inclus** dans ce repo.

## 🛠️ Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## 📄 Licence

MIT — usage éducatif.
