import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*60)
print("🔄 ENTRAÎNEMENT DU MODÈLE")
print("="*60)

# Charger les données
print("\n1. Chargement des données...")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(f"   ✅ {len(X)} échantillons chargés")

# Split train/test
print("\n2. Séparation train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   ✅ Train: {len(X_train)} | Test: {len(X_test)}")

# Normalisation
print("\n3. Normalisation...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Normalisation effectuée")

# Entraînement
print("\n4. Entraînement du modèle...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("   ✅ Modèle entraîné")

# Évaluation
print("\n5. Évaluation...")
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n   📊 MÉTRIQUES:")
print(f"   ├─ MAE  : {mae:.2f}")
print(f"   ├─ RMSE : {rmse:.2f}")
print(f"   └─ R²   : {r2:.4f}")

# Sauvegarde
print("\n6. Sauvegarde...")
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'diabetes_scaler.pkl')
print("   ✅ diabetes_model.pkl")
print("   ✅ diabetes_scaler.pkl")

print("\n" + "="*60)
print("✨ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*60)
print("\n💡 Lancez maintenant: streamlit run app.py\n")