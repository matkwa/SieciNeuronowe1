import numpy as np
import pandas as pd
import time
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# =========================================================
# 1. DANE
# =========================================================
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================================
# 2. OVERSAMPLING
# =========================================================
train_df = pd.DataFrame(X_train)
train_df['income'] = y_train

df_major = train_df[train_df.income == 0]
df_minor = train_df[train_df.income == 1]

df_minor_up = resample(
    df_minor,
    replace=True,
    n_samples=len(df_major),
    random_state=42
)

train_balanced = pd.concat([df_major, df_minor_up]).sample(frac=1)

X_train = train_balanced.drop('income', axis=1).values
y_train = train_balanced['income'].values


# =========================================================
# 3. PRZESTRZEŃ HIPERPARAMETRÓW
# =========================================================
architectures = [
    [64, 32],
    [128, 64],
    [128, 64, 32]
]

dropouts = [0.2, 0.3, 0.4]
learning_rates = [0.001, 0.0005]

search_space = list(product(architectures, dropouts, learning_rates))


# =========================================================
# 4. FUNKCJA BUDUJĄCA MODEL
# =========================================================
def build_model(layers, dropout, lr):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))

    for units in layers:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# =========================================================
# 5. AUTO-SEARCH
# =========================================================
best_auc = 0
best_config = None
trial = 1

start_all = time.time()

for layers, dropout, lr in search_space:
    print(f"\n🔍 Trial {trial}/{len(search_space)}")
    print(f"   Layers={layers}, Dropout={dropout}, LR={lr}")

    tf.keras.backend.clear_session()
    model = build_model(layers, dropout, lr)

    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early],
        verbose=0
    )

    probs = model.predict(X_test, verbose=0).ravel()
    auc = roc_auc_score(y_test, probs)

    print(f"   AUC = {auc:.4f}")

    if auc > best_auc:
        print("   🔥 NEW BEST MODEL")
        best_auc = auc
        best_config = (layers, dropout, lr)
        model.save('best_automl_adult.keras')

    trial += 1


# =========================================================
# 6. PODSUMOWANIE
# =========================================================
total_time = time.time() - start_all

print("\n" + "="*60)
print("🏆 NAJLEPSZY MODEL")
print(f"AUC: {best_auc:.4f}")
print(f"Layers: {best_config[0]}")
print(f"Dropout: {best_config[1]}")
print(f"Learning Rate: {best_config[2]}")
print(f"⏱️ Czas: {total_time/60:.1f} min")
print("💾 Zapisano: best_automl_adult.keras")
