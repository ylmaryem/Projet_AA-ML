import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from joblib import load

# Configuration de la page Streamlit
st.set_page_config(page_title="Tableau de Bord des Attaques", layout="wide")

# Charger le modèle sauvegardé
model = load('random_forest_model.joblib')

# Fonction de prétraitement
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)
    
    # Retirer les colonnes inutiles
    df = df.drop(['id', 'proto', 'state', 'service'], axis=1, errors='ignore')
    
    # Remplacer les tirets par NaN
    df.replace('-', np.nan, inplace=True)
    
    # Sélectionner les colonnes numériques et catégorielles
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    # Remplir les NaN
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

    # Division des données
    X = df.drop(['attack_cat', 'label'], axis=1, errors='ignore')
    y = df['attack_cat'] if 'attack_cat' in df.columns else df['label']
    
    return df, X, y

# Interface utilisateur principale
st.title("🔥 Dashboard de Détection des Attaques")

# Barre latérale pour les options
with st.sidebar:
    st.header("⚙ Options de Configuration")
    uploaded_file = st.file_uploader("Chargez un fichier CSV", type=["csv"])

# Si un fichier est chargé
if uploaded_file:
    df, X, y = load_and_preprocess(uploaded_file)
    y_pred = model.predict(X)

    # Onglets pour organiser le contenu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Aperçu des Données",  "❌ Erreurs et Prédictions", "📊 Répartition des Attaques Détectées",  "🔍 Importance des Caractéristiques", "📊 Métriques d'Évaluation du Modèle"
    ])
    
    # Onglet 1 : Aperçu des Données
    with tab1:
        st.subheader("📋 Aperçu des Données")
        st.dataframe(df)
        st.write("Dimensions des données :", df.shape)
        st.subheader("📊 Statistiques Descriptives")
        st.dataframe(df.describe())

      # Onglet 2 : Erreurs et Prédictions
    with tab2:
        # Afficher les erreurs (échantillons mal prédits)
        st.header("Échantillons mal prédits")
        erreurs = df[y != y_pred]  # Extraire les échantillons mal prédits
        erreurs['Prédiction'] = y_pred[y != y_pred]  # Ajouter la colonne de prédiction
        erreurs['Vraie étiquette'] = y[y != y_pred]  # Ajouter la colonne de vraie étiquette
        st.dataframe(erreurs)

        # Exploration des prédictions individuelles
        st.header("Prédictions individuelles")
        index = st.number_input("Index de l'échantillon (0 à {})".format(len(X) - 1), 0, len(X) - 1, 0)
        
        # Extraire l'échantillon
        sample = X.iloc[[index]]  # Double crochets pour garder un DataFrame
        st.write("Caractéristiques de l'échantillon :", sample)
        
        # Faire la prédiction pour cet échantillon
        predicted_label = model.predict(sample)[0]  # Prédiction
        st.write("Prédiction :", predicted_label)
        
        # Obtenir la vraie étiquette de l'échantillon
        true_label = y.iloc[index]  # Récupérer l'étiquette correcte
        st.write("Vraie étiquette :", true_label)
        
        # Vérifier si la prédiction est correcte
        if predicted_label == true_label:
            st.success("La prédiction est correcte !")
        else:
            st.error("La prédiction est incorrecte.")

    # Onglet 3 : Statistiques des Attaques
    with tab3:
        st.subheader("📊 Répartition des Attaques Détectées")
        # Compter les attaques prédites
        attack_counts = pd.Series(y_pred).value_counts()
        
        # Convertir en DataFrame pour renommer les colonnes
        attack_counts_df = attack_counts.reset_index()
        attack_counts_df.columns = ['Type d\'attaque', 'Nombre d\'occurrences']
        
        # Affichage du graphique
        st.bar_chart(attack_counts)
        
        # Affichage des détails des attaques avec des noms de colonnes modifiés
        st.write("📝 Détail des Attaques")
        st.dataframe(attack_counts_df, use_container_width=True)


    
    # Onglet 4 : Importance des Caractéristiques
    with tab4:
        st.subheader("🔍 Importance des Caractéristiques")
        feature_importances = model.feature_importances_
        features = X.columns

        importance_df = pd.DataFrame({
            "Caractéristique": features,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Caractéristique"))
        

    # Onglet 5 : Métriques d'Évaluation du Modèle
    with tab5:
        st.subheader("📊 Métriques d'Évaluation du Modèle")
        
        # Afficher les métriques d'évaluation sous forme de tableau
        st.write("📝 Rapport de Classification ")
        report = classification_report(y, y_pred, output_dict=True)
        
        # Conversion du rapport en DataFrame pour une présentation propre
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Affichage des métriques globales
        st.write("Précision Globale du Modèle :", round(report['accuracy'], 3))
        st.write("Score F1 Moyen (Macro) :", round(report['macro avg']['f1-score'], 3))
        
        # Ajouter une section graphique combinée
        col1, col2 = st.columns(2)  # Crée deux colonnes côte à côte
    
        # Colonne 1 : Matrice de confusion
        with col1:
            st.write("🗺 Matrice de Confusion")
            cm = confusion_matrix(y, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
            ax_cm.set_xlabel("Prédictions")
            ax_cm.set_ylabel("Vérités")
            st.pyplot(fig_cm)
    
        # Colonne 2 : Courbe ROC
        with col2:
            st.write("📈 Courbe ROC")
            
            # Calcul des probabilités et des métriques ROC
            y_prob = model.predict_proba(X)[:, 1]  # Probabilités pour la classe positive
            fpr, tpr, thresholds = roc_curve(y, y_prob, pos_label=model.classes_[1])
            roc_auc = auc(fpr, tpr)
            
            # Tracer la courbe ROC
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color="blue")
            ax_roc.plot([0, 1], [0, 1], 'r--', label="Aléatoire")
            ax_roc.set_xlabel("Taux de Faux Positifs (FPR)")
            ax_roc.set_ylabel("Taux de Vrais Positifs (TPR)")
            ax_roc.set_title("Courbe ROC")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

# Si aucun fichier n'est chargé
else:
    st.info("📂 Veuillez charger un fichier CSV pour commencer.")



# Pour voir le dashboard, terminal->taper la commande: streamlit run dashboard.py