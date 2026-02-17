# 🔒 Projet Détection des Attaques (IDS)

## Description
Projet de détection des attaques réseau avec un modèle **Random Forest** et un **dashboard interactif Streamlit**.

## Fonctionnalités
- Prétraitement des données (valeurs manquantes, colonnes, encodage)
- Rééquilibrage des classes avec **SMOTE**
- Entraînement d’un modèle **Random Forest**
- Dashboard Streamlit pour visualiser :
  - Aperçu des données 📋  
  - Prédictions et erreurs ❌  
  - Répartition des attaques 📊  
  - Importance des caractéristiques 🔍  
  - Métriques d’évaluation 📈

## Utilisation
1. Installer les dépendances :  
```bash
pip install -r requirements.txt
```
2. Lancer le dashboard : 
```bash
streamlit run dashboard.py
```
3. Charger un fichier CSV pour explorer et prédire.
