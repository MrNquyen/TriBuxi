import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('data/final_data.csv') 
# Convert columns to 1/0
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

df['ADHERENCE'] = (
    df['ADHERENCE']
    .replace({'ADHERENT': 1, 'NON-ADHERENT': 0})
    .infer_objects(copy=False)
)

df_model_RF_2 = df.copy()

features_model = [
    "AGE",
    "ANNUALCLAIMAMOUNT",
    "UNITSTOTAL"
]

X = df_model_RF_2[features_model]
y = df_model_RF_2["ADHERENCE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf_model_2 = RandomForestClassifier(
    max_depth=7,
    n_estimators=200,
    criterion='entropy',
    random_state=1
)
rf_model_2.fit(X_train, y_train)

joblib.dump(rf_model_2, 'save/rf_model.pkl')

y_pred = rf_model_2.predict(X_test)
y_proba = rf_model_2.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model (3 features)")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Adherent', 'Adherent'])
disp.plot(cmap="Blues")
plt.savefig('save/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()