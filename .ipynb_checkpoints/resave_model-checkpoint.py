import joblib

# Étape 1 : Charger l'ancien modèle
try:
    model = joblib.load("diabetes_model.pkl")
    print("✅ Ancien modèle chargé avec succès")
except FileNotFoundError:
    print("❌ Fichier 'diabetes_model.pkl' introuvable")
    exit()
except ModuleNotFoundError as e:
    print(f"❌ Module manquant : {e}")
    exit()

# Étape 2 : Sauvegarder le modèle dans un nouveau fichier
try:
    joblib.dump(model, "diabetes_model_new.pkl")
    print("✅ Modèle resauvegardé pour la version actuelle")
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde : {e}")
    exit()

# Étape 3 : Tester le modèle resauvegardé
try:
    model_test = joblib.load("diabetes_model_new.pkl")
    print("✅ Nouveau modèle chargé correctement")
except Exception as e:
    print(f"❌ Erreur lors du test du nouveau modèle : {e}")
