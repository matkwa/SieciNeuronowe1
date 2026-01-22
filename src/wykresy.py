import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             classification_report, accuracy_score, 
                             precision_score, recall_score, f1_score)
from sklearn.utils import resample
import tensorflow as tf

# =========================================================
# KONFIGURACJA
# =========================================================
MODEL_FILENAME = 'best_automl_adult.keras' # Sprawdź nazwę!
DATA_FILENAME = 'adult.csv'
RESULTS_FILENAME = 'wyniki.txt' # Tu zapiszemy liczby

# =========================================================
# 1. PRZYGOTOWANIE DANYCH
# =========================================================
print("⏳ Ładowanie i przetwarzanie danych...")
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
           'marital.status', 'occupation', 'relationship', 'race', 'sex',
           'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income']

df = pd.read_csv(DATA_FILENAME, names=columns, na_values='?', skipinitialspace=True)
df = df.dropna()
df = df.drop(['fnlwgt', 'education'], axis=1)
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('income', axis=1).values
y = df['income'].values

# Split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_raw = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Balansowanie testu 50/50
print("⚖️  Balansowanie zbioru testowego do 50/50...")
test_df = pd.DataFrame(X_test)
test_df['y'] = y_test_raw

df_major = test_df[test_df.y == 0]
df_minor = test_df[test_df.y == 1]

df_major_down = resample(df_major, replace=False, n_samples=len(df_minor), random_state=42)
test_balanced = pd.concat([df_major_down, df_minor]).sample(frac=1, random_state=42)

X_test_final = test_balanced.drop('y', axis=1).values
y_test_final = test_balanced['y'].values

print(f"✅ Dane gotowe. Test set size: {len(y_test_final)}")

# =========================================================
# 2. ŁADOWANIE MODELU I PREDYKCJE
# =========================================================
print(f"📂 Ładowanie modelu: {MODEL_FILENAME}...")
try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
except Exception as e:
    print(f"❌ BŁĄD: Nie znaleziono pliku modelu lub zła nazwa! ({e})")
    exit()

print("🔮 Obliczanie predykcji...")
y_pred_probs = model.predict(X_test_final, verbose=0).ravel()
y_pred = (y_pred_probs > 0.5).astype(int)

# =========================================================
# 3. OBLICZANIE METRYK I ZAPIS DO PLIKU
# =========================================================
print(f"📝 Obliczanie metryk i zapis do {RESULTS_FILENAME}...")

# Oblicz metryki
acc = accuracy_score(y_test_final, y_pred)
prec = precision_score(y_test_final, y_pred)
rec = recall_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)

# Oblicz AUC
fpr, tpr, thresholds = roc_curve(y_test_final, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Macierz pomyłek
cm = confusion_matrix(y_test_final, y_pred)
tn, fp, fn, tp = cm.ravel()

# Zapisz do pliku
with open(RESULTS_FILENAME, 'w', encoding='utf-8') as f:
    f.write("=== SZCZEGÓŁOWE WYNIKI MODELU ===\n")
    f.write(f"Model: {MODEL_FILENAME}\n\n")
    
    f.write(f"ACCURACY (Dokładność): {acc:.4f}\n")
    f.write(f"AUC (Area Under Curve): {roc_auc:.4f}\n")
    f.write(f"PRECISION (Precyzja dla klasy >50K): {prec:.4f}\n")
    f.write(f"RECALL (Czułość dla klasy >50K): {rec:.4f}\n")
    f.write(f"F1-SCORE: {f1:.4f}\n\n")
    
    f.write("--- MACIERZ POMYŁEK ---\n")
    f.write(f"True Negatives (TN): {tn}\n")
    f.write(f"False Positives (FP): {fp}\n")
    f.write(f"False Negatives (FN): {fn}\n")
    f.write(f"True Positives (TP): {tp}\n\n")
    
    f.write("--- RAPORT KLASYFIKACJI ---\n")
    f.write(classification_report(y_test_final, y_pred))

print(f"💾 Zapisano wyniki w pliku: {RESULTS_FILENAME}")

# =========================================================
# 4. GENEROWANIE WYKRESÓW
# =========================================================

# --- A. Macierz Pomyłek ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, annot_kws={"size": 16})
plt.title('Macierz Pomyłek (Confusion Matrix)', fontsize=16)
plt.xlabel('Przewidziana klasa', fontsize=12)
plt.ylabel('Prawdziwa klasa', fontsize=12)
plt.xticks([0.5, 1.5], ['<=50K', '>50K'], fontsize=12)
plt.yticks([0.5, 1.5], ['<=50K', '>50K'], fontsize=12, rotation=0)
plt.tight_layout()
plt.savefig('macierz.png', dpi=300)
print("💾 Zapisano: macierz.png")
plt.close()

# --- B. Krzywa ROC ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specyficzność)', fontsize=12)
plt.ylabel('True Positive Rate (Czułość)', fontsize=12)
plt.title('Krzywa ROC', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc.png', dpi=300)
print("💾 Zapisano: roc.png")
plt.close()
