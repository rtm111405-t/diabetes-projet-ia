import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Chargement dataset Diabetes (10 features normalisées)
print("Chargement dataset Diabetes...")
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et fit du scaler
print("Création scaler_new...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement modèle
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Évaluation rapide
y_pred = model.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(np.mean((y_test - y_pred)**2)):.2f}")
print(f"R²: {model.score(X_test_scaled, y_test):.3f}")

# Sauvegarde scaler ET modèle
joblib.dump(scaler, 'scaler_new.pkl')
joblib.dump(model, 'diabetes_model_new.pkl')
print("✅ Fichiers créés : scaler_new.pkl et diabetes_model_new.pkl")
