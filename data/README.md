# Données

Le dataset **n'est pas inclus** dans ce repo (données Kaggle).

## Téléchargement

1. Aller sur : https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection
2. Télécharger `bank_transactions_data_2.csv`
3. Placer le fichier dans ce dossier (`data/`)

## Schéma

| Colonne | Description |
|---|---|
| `TransactionID` | ID unique |
| `AccountID` | Compte |
| `TransactionAmount` | Montant |
| `TransactionDate` / `PreviousTransactionDate` | Horodatage |
| `TransactionType` | Credit / Debit |
| `AccountBalance` | Solde **post**-transaction |
| `Location`, `DeviceID`, `IP Address`, `MerchantID` | Contexte |
| `Channel` | Online / ATM / Branch |
| `CustomerAge`, `CustomerOccupation` | Démographie |
| `TransactionDuration`, `LoginAttempts` | Comportement |
