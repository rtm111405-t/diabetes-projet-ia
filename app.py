"""
Application Streamlit pour la prédiction de progression du diabète
Mini-Projet IA - Groupe 2
"""

import streamlit as st
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Diabète",
    page_icon="🏥",
    layout="wide"
)

# Fonction de chargement du modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/diabetes_model_new.pkl')
        scaler = joblib.load('models/diabetes_scaler_new.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Fichiers modèle introuvables. Veuillez d'abord entraîner le modèle avec train.py")
        st.stop()

# Fonction pour normaliser l'âge (supposons que l'âge va de 20 à 80 ans)
def normalize_age(age_reel):
    # Normalisation basée sur la distribution du dataset diabetes
    # Moyenne ~48 ans, écart-type ~13 ans
    age_norm = (age_reel - 48.5) / 13.1
    return age_norm

# Fonction pour convertir sexe en valeur normalisée
def normalize_sex(sexe):
    # Dans le dataset: Femme = -0.044, Homme = +0.051 (environ)
    return 0.051 if sexe == "Homme" else -0.044

# Fonction pour dénormaliser l'âge
def denormalize_age(age_norm):
    return age_norm * 13.1 + 48.5


# Charger le modèle
model, scaler = load_model()

# Titre principal
st.title("🏥 Prédiction Progression Diabète 2025_2026")
st.markdown("---")

# ========== SIDEBAR AVEC TESTS RAPIDES ==========
st.sidebar.header("🧪 Tests Rapides")
st.sidebar.markdown("Cliquez pour charger des valeurs de test :")

if st.sidebar.button("📊 TEST 1 : Patient Normal", use_container_width=True):
    st.session_state.age_reel = 50
    st.session_state.sexe = "Homme"
    st.session_state.bmi = 0.062
    st.session_state.bp = 0.022
    st.session_state.s1 = -0.005
    st.session_state.s2 = -0.008
    st.session_state.s3 = -0.004
    st.session_state.s4 = -0.002
    st.session_state.s5 = 0.003
    st.session_state.s6 = 0.018
    st.sidebar.success("✅ Valeurs chargées ! Score attendu: ~150")

if st.sidebar.button("🟢 TEST 2 : Faible Progression", use_container_width=True):
    st.session_state.age_reel = 40
    st.session_state.sexe = "Homme"
    st.session_state.bmi = -0.040
    st.session_state.bp = -0.050
    st.session_state.s1 = -0.060
    st.session_state.s2 = -0.080
    st.session_state.s3 = 0.050
    st.session_state.s4 = -0.040
    st.session_state.s5 = -0.070
    st.session_state.s6 = -0.050
    st.sidebar.success("✅ Valeurs chargées ! Score attendu: ~70")

if st.sidebar.button("🔴 TEST 3 : Haute Progression", use_container_width=True):
    st.session_state.age_reel = 58
    st.session_state.sexe = "Homme"
    st.session_state.bmi = 0.120
    st.session_state.bp = 0.080
    st.session_state.s1 = 0.090
    st.session_state.s2 = 0.100
    st.session_state.s3 = -0.080
    st.session_state.s4 = 0.050
    st.session_state.s5 = 0.150
    st.session_state.s6 = 0.120
    st.sidebar.success("✅ Valeurs chargées ! Score attendu: ~280")

if st.sidebar.button("⚪ TEST 4 : Valeurs Moyennes", use_container_width=True):
    st.session_state.age_reel = 48
    st.session_state.sexe = "Femme"
    st.session_state.bmi = 0.0
    st.session_state.bp = 0.0
    st.session_state.s1 = 0.0
    st.session_state.s2 = 0.0
    st.session_state.s3 = 0.0
    st.session_state.s4 = 0.0
    st.session_state.s5 = 0.0
    st.session_state.s6 = 0.0
    st.sidebar.success("✅ Valeurs chargées ! Score attendu: ~152")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Astuce:** Cliquez sur un test, puis sur 'PRÉDIRE' en bas")

# ========== INTERFACE PRINCIPALE ==========
st.write("Saisissez les valeurs des 10 variables et cliquez sur **Prédire**.")

# Création des inputs avec session_state
col1, col2 = st.columns(2)

with col1:
    age_reel = st.number_input(
        "Age (années)", 
        min_value=18, 
        max_value=100, 
        value=st.session_state.get('age_reel', 48),
        step=1,
        help="Entrez l'âge réel du patient en années"
    )
    
    sexe = st.selectbox(
        "Sexe",
        options=["Homme", "Femme"],
        index=0 if st.session_state.get('sexe', 'Homme') == "Homme" else 1,
        help="Sélectionnez le sexe du patient"
    )
    
    bmi = st.number_input("BMI (Indice de Masse Corporelle)", value=st.session_state.get('bmi', 0.0), step=0.01, format="%.3f")
    bp = st.number_input("Pression artérielle (bp)", value=st.session_state.get('bp', 0.0), step=0.01, format="%.3f")
    s1 = st.number_input("s1 (mesure sanguine 1)", value=st.session_state.get('s1', 0.0), step=0.01, format="%.3f")

with col2:
    s2 = st.number_input("s2 (mesure sanguine 2)", value=st.session_state.get('s2', 0.0), step=0.01, format="%.3f")
    s3 = st.number_input("s3 (mesure sanguine 3)", value=st.session_state.get('s3', 0.0), step=0.01, format="%.3f")
    s4 = st.number_input("s4 (mesure sanguine 4)", value=st.session_state.get('s4', 0.0), step=0.01, format="%.3f")
    s5 = st.number_input("s5 (mesure sanguine 5)", value=st.session_state.get('s5', 0.0), step=0.01, format="%.3f")
    s6 = st.number_input("s6 (mesure sanguine 6)", value=st.session_state.get('s6', 0.0), step=0.01, format="%.3f")

st.markdown("---")

# Bouton de prédiction
if st.button("🔮 PRÉDIRE", use_container_width=True, type="primary"):
    # Normaliser l'âge et le sexe
    age_norm = normalize_age(age_reel)
    sex_norm = normalize_sex(sexe)
    
    # Préparer les données
    input_data = np.array([[age_norm, sex_norm, bmi, bp, s1, s2, s3, s4, s5, s6]])
    input_scaled = scaler.transform(input_data)
    
    # Faire la prédiction
    prediction = model.predict(input_scaled)[0]
    
    # Afficher le résultat avec style
    st.markdown("---")
    st.subheader("📊 Résultat de la Prédiction")
    
    # Affichage des informations du patient
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info(f"👤 **Âge:** {age_reel} ans")
    with col_info2:
        st.info(f"⚧ **Sexe:** {sexe}")
    with col_info3:
        st.info(f"📊 **BMI:** {bmi:.3f}")
    
    st.markdown("###")
    
    # Affichage principal du score
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.metric(label="Score de Progression Prédit", value=f"{prediction:.2f}")
    
    # Interprétation avec couleurs
    if prediction < 100:
        st.success("✅ **Interprétation: FAIBLE PROGRESSION**")
        st.info("Le patient présente une progression lente de la maladie.")
    elif prediction < 150:
        st.warning("⚠️ **Interprétation: PROGRESSION MODÉRÉE**")
        st.info("Le patient présente une progression moyenne de la maladie.")
    elif prediction < 200:
        st.warning("🟠 **Interprétation: PROGRESSION ÉLEVÉE**")
        st.info("Le patient présente une progression rapide de la maladie.")
    else:
        st.error("🔴 **Interprétation: PROGRESSION TRÈS ÉLEVÉE**")
        st.info("Le patient présente une progression très rapide de la maladie.")
    
    # Informations complémentaires
    st.markdown("---")
    st.caption("💡 **Note:** La valeur typique se situe entre 0 et 300+ (plus élevé = progression plus rapide)")

# Mise à jour des session_state
st.session_state.age_reel = age_reel
st.session_state.sexe = sexe
st.session_state.bmi = bmi
st.session_state.bp = bp
st.session_state.s1 = s1
st.session_state.s2 = s2
st.session_state.s3 = s3
st.session_state.s4 = s4
st.session_state.s5 = s5
st.session_state.s6 = s6

# Footer
st.markdown("---")
st.caption("🏥 Système de Prédiction du Diabète | Mini-Projet IA 2025_2026")