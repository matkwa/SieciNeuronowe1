import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# KONFIGURACJA (Parametry zwycięskiego modelu)
BEST_PARAMS = {
    'layers': [64, 32],
    'dropout': 0.3,
    'lr': 0.0005,
    'batch_size': 32,
    'epochs': 60
}

# 1. PRZYGOTOWANIE DANYCH (Standardowa procedura)
print("⏳ Przygotowanie danych do wizualizacji procesu uczenia...")
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
           'marital.status', 'occupation', 'relationship', 'race', 'sex',
           'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income']

df = pd.read_csv('adult.csv', names=columns, na_values='?', skipinitialspace=True).dropna()
df = df.drop(['fnlwgt', 'education'], axis=1)
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('income', axis=1).values
y = df['income'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Oversampling treningowy
train_df = pd.DataFrame(X_train)
train_df['y'] = y_train
df_maj = train_df[train_df.y == 0]
df_min = train_df[train_df.y == 1]
df_min_up = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
train_bal = pd.concat([df_maj, df_min_up]).sample(frac=1, random_state=42)
X_train_bal = train_bal.drop('y', axis=1).values
y_train_bal = train_bal['y'].values

# Scaling
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
# (Testu nie potrzebujemy do wykresu loss, ale dla formalności walidacja jest robiona na fragmencie train)

# 2. BUDOWA MODELU (Odtworzenie najlepszej architektury)
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))

for units in BEST_PARAMS['layers']:
    model.add(Dense(units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(BEST_PARAMS['dropout']))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=BEST_PARAMS['lr']),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. TRENING Z ZAPISEM HISTORII
print("🚀 Uruchamianie treningu demonstracyjnego...")
history = model.fit(
    X_train_bal, y_train_bal,
    epochs=BEST_PARAMS['epochs'],
    batch_size=BEST_PARAMS['batch_size'],
    validation_split=0.2, # Wydzielenie walidacji żeby pokazać linię val_loss
    callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
    verbose=1
)

# 4. RYSOWANIE WYKRESU
print("🎨 Generowanie wykresu...")
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(14, 5))

# Wykres 1: LOSS
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Trening Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Walidacja Loss', linewidth=2, linestyle='--')
plt.title('Krzywa uczenia (Funkcja Straty)', fontsize=14)
plt.xlabel('Epoki')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Wykres 2: ACCURACY
plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='Trening Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Walidacja Accuracy', linewidth=2, linestyle='--')
plt.title('Krzywa uczenia (Dokładność)', fontsize=14)
plt.xlabel('Epoki')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300)
print("💾 Zapisano: learning_curve.png")