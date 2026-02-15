import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create directory
os.makedirs("models", exist_ok=True)

print("Step 1: Loading heart.csv...")
df = pd.read_csv("heart.csv")

# 1. Handle Categorical Columns
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    # Save as lowercase to avoid Windows issues
    joblib.dump(le, f"models/{col.lower()}_encoder.pkl")
    print(f" - Saved: models/{col.lower()}_encoder.pkl")

# 2. Split Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")
print(" - Saved: models/scaler.pkl")

# 4. Train 6 Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = []
for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds
    
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds)
    })
    
    # Save with lowercase name
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(clf, f"models/{filename}")
    print(f" - Saved: models/{filename}")

# 5. Save Results Table
pd.DataFrame(results).to_csv("model_results.csv", index=False)
print("\n✅ TRAINING SUCCESSFUL. All files are in the 'models' folder.")