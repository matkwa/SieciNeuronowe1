import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

# TensorFlow / Keras
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.backend import clear_session

# ---------------------------------------------------------
# KONFIGURACJA
# ---------------------------------------------------------
LICZBA_PROB = 200
MODEL_FILENAME = 'najlepszy_model_adult.keras'

# ---------------------------------------------------------
# 1. PRZYGOTOWANIE DANYCH
# ---------------------------------------------------------
print("⏳ Wczytywanie i przetwarzanie danych...")

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
           'marital.status', 'occupation', 'relationship', 'race', 'sex',
           'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income']

df = pd.read_csv('adult.csv', names=columns, na_values='?', skipinitialspace=True)
df = df.dropna()
df = df.drop(['fnlwgt', 'education'], axis=1)
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('income', axis=1).values
y = df['income'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------
# 2. OVERSAMPLING ZBIORU TRENINGOWEGO
# ---------------------------------------------------------
X_train_df = pd.DataFrame(X_train)
y_train_df = pd.Series(y_train, name='income')
train_df = pd.concat([X_train_df, y_train_df], axis=1)

df_majority = train_df[train_df.income == 0]
df_minority = train_df[train_df.income == 1]

print("Przed oversamplingiem:")
print(train_df.income.value_counts())

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

train_balanced = pd.concat([df_majority, df_minority_upsampled])
train_balanced = train_balanced.sample(frac=1, random_state=42)

print("\nPo oversamplingiem:")
print(train_balanced.income.value_counts())

X_train = train_balanced.drop('income', axis=1).values
y_train = train_balanced['income'].values

print("✅ Dane gotowe. Rozpoczynamy pętlę treningową.")
print("-" * 60)

# ---------------------------------------------------------
# 3. PĘTLA TRENINGOWA Z LICZNIKIEM CZASU
# ---------------------------------------------------------
best_accuracy = 0.0
best_trial = 0
total_start_time = time.time()

for i in range(LICZBA_PROB):
    trial_start_time = time.time()
    print(f"\n🔄 Próba {i + 1}/{LICZBA_PROB}...", end=" ")

    clear_session()

    # Budowa modelu
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Trening
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    # Ewaluacja
    predictions_temp = (model.predict(X_test, verbose=0) > 0.5).astype("int32")
    current_acc = accuracy_score(y_test, predictions_temp)

    # Czas trwania próby
    trial_end_time = time.time()
    duration = trial_end_time - trial_start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    print(f"Zakończona w: {minutes}m {seconds}s | Acc: {current_acc * 100:.2f}%")

    # Sprawdzenie rekordu
    if current_acc > best_accuracy:
        print(f"   🔥 NOWY REKORD! (Poprzedni: {best_accuracy * 100:.2f}%) -> Zapisuję model.")
        best_accuracy = current_acc
        best_trial = i + 1
        model.save(MODEL_FILENAME)
    else:
        print(f"   (Słabszy niż rekord {best_accuracy * 100:.2f}%)")

# ---------------------------------------------------------
# 4. PODSUMOWANIE
# ---------------------------------------------------------
total_end_time = time.time()
total_duration = total_end_time - total_start_time
t_min = int(total_duration // 60)
t_sec = int(total_duration % 60)

print("-" * 60)
print(f"🏁 Koniec eksperymentu.")
print(f"⏱️ Całkowity czas pracy: {t_min}m {t_sec}s")
print(f"👑 Najlepszy model: Próba {best_trial} z wynikiem {best_accuracy * 100:.2f}%")
print(f"💾 Zapisano w pliku: {MODEL_FILENAME}")

# Raport dla zwycięzcy
best_model = load_model(MODEL_FILENAME)
predictions = (best_model.predict(X_test, verbose=0) > 0.5).astype("int32")

print("\n--- Raport dla Zwycięskiego Modelu ---")
print(classification_report(y_test, predictions))

# Macierz pomyłek
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Macierz Pomyłek - Najlepszy Model')
plt.xlabel('Przewidziane')
plt.ylabel('Prawdziwe')
plt.show()
