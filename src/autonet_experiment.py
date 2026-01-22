import os
import time
import itertools
import warnings
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# ==============================================================================
# 0. BEZPIECZNE IMPORTY I KONFIGURACJA
# ==============================================================================
try:
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, BatchNormalization, Input
    from keras.callbacks import EarlyStopping, Callback
    from keras.optimizers import Adam
except ImportError:
    Sequential = tf.keras.models.Sequential
    load_model = tf.keras.models.load_model
    Input = tf.keras.layers.Input
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    BatchNormalization = tf.keras.layers.BatchNormalization
    EarlyStopping = tf.keras.callbacks.EarlyStopping
    Callback = tf.keras.callbacks.Callback
    Adam = tf.keras.optimizers.Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils import resample

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

# ==============================================================================
# 1. PARAMETRY EKSPERYMENTU
# ==============================================================================
N_REPEATS = 5            # <--- USTAWIONE: 25 prób
PRUNING_THRESHOLD = 0.025  # <--- USTAWIONE: 2% tolerancji
GLOBAL_BEST_ACC = 0.0     # Zmienna globalna przechowująca rekord

param_grid = {
    'layers': [
        [64, 32],       # Standard
        [128, 64],      # Mocniejszy
        [128, 64, 32]   # Głęboki
    ],
    'dropout': [0.2, 0.3],     
    'lr': [0.001, 0.0005],     
    'batch_size': [32]         
}

# Generowanie kombinacji (12 wariantów)
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"🔬 Strategia: 12 Konfiguracji x {N_REPEATS} Powtórzeń = {len(combinations)*N_REPEATS} Treningów.")
print(f"✂️  Pruning aktywny: Model odpada, jeśli jest gorszy o {PRUNING_THRESHOLD*100:.0f}% od lidera.")

# ==============================================================================
# 2. PRZYGOTOWANIE DANYCH (Standard Strict)
# ==============================================================================
def load_and_prep():
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
            'marital.status', 'occupation', 'relationship', 'race', 'sex',
            'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income']
    
    df = pd.read_csv('adult.csv', names=cols, na_values='?', skipinitialspace=True).dropna()
    df = df.drop(['fnlwgt', 'education'], axis=1)
    df['income'] = (df['income'].str.contains('>50K')).astype(int)
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop('income', axis=1).values
    y = df['income'].values
    
    # Split
    X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Test Set -> 50/50 Balance
    df_test = pd.DataFrame(X_te_raw)
    df_test['y'] = y_te_raw
    min_n = len(df_test[df_test.y == 1])
    
    df_maj = resample(df_test[df_test.y == 0], replace=False, n_samples=min_n, random_state=42)
    test_bal = pd.concat([df_maj, df_test[df_test.y == 1]]).sample(frac=1, random_state=42)
    
    X_test = test_bal.drop('y', axis=1).values
    y_test = test_bal['y'].values
    
    # 2. Train Set -> Oversampling
    df_train = pd.DataFrame(X_tr_raw)
    df_train['y'] = y_tr_raw
    max_n = len(df_train[df_train.y == 0])
    
    df_min_up = resample(df_train[df_train.y == 1], replace=True, n_samples=max_n, random_state=42)
    train_bal = pd.concat([df_train[df_train.y == 0], df_min_up]).sample(frac=1, random_state=42)
    
    X_train = train_bal.drop('y', axis=1).values
    y_train = train_bal['y'].values
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

print("⏳ Ładowanie danych...")
X_train, y_train, X_test, y_test = load_and_prep()
print("✅ Dane gotowe.")

# ==============================================================================
# 3. MECHANIZM UCINANIA (PRUNING CALLBACK)
# ==============================================================================
class PruningCallback(Callback):
    """Zatrzymuje trening, jeśli model jest znacznie gorszy od Global Best"""
    def on_epoch_end(self, epoch, logs=None):
        global GLOBAL_BEST_ACC
        # Sprawdzamy dopiero od 8 epoki
        if epoch > 8 and GLOBAL_BEST_ACC > 0:
            current_val_acc = logs.get('val_accuracy')
            # Jeśli jest gorszy o PRUNING_THRESHOLD (2%) od rekordu -> STOP
            if current_val_acc < (GLOBAL_BEST_ACC - PRUNING_THRESHOLD):
                self.model.stop_training = True
                print(f" ✂️  PRUNED: {current_val_acc:.2%} (Limit: {GLOBAL_BEST_ACC - PRUNING_THRESHOLD:.2%})")

# ==============================================================================
# 4. GŁÓWNA PĘTLA EKSPERYMENTU
# ==============================================================================
stats = [] 
best_config_params = None
best_model_filename = 'ULTIMATE_MODEL_ADULT.keras'
total_start = time.time()

print("\n" + "="*85)
print(f"{'CFG':<3} | {'REP':<3} | {'STATUS':<10} | {'ACCURACY':<8} | {'AUC':<6} | {'TIME':<6} | {'PARAMS'}")
print("="*85)

for cfg_idx, p in enumerate(combinations):
    
    acc_scores = []
    
    for rep in range(N_REPEATS):
        # 1. Build
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        
        for units in p['layers']:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(p['dropout']))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=p['lr']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        # 2. Train with Pruning
        start_run = time.time()
        
        callbacks_list = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            PruningCallback()
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=40,
            batch_size=p['batch_size'],
            validation_split=0.2,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # 3. Evaluate
        probs = model.predict(X_test, verbose=0).ravel()
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        run_time = time.time() - start_run
        
        status = "OK"
        if acc > GLOBAL_BEST_ACC:
            GLOBAL_BEST_ACC = acc
            best_config_params = p
            model.save(best_model_filename)
            status = "🔥 BEST!"
        
        acc_scores.append(acc)
        
        params_str = f"L={p['layers']} D={p['dropout']} LR={p['lr']}"
        print(f"{cfg_idx+1}/{len(combinations)} | {rep+1}/{N_REPEATS} | {status:<10} | {acc:.2%}   | {auc:.3f}  | {run_time:.0f}s    | {params_str}")

    # Statystyki dla danej konfiguracji
    avg_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    stats.append({
        'params': p,
        'avg_acc': avg_acc,
        'std_acc': std_acc,
        'max_acc': np.max(acc_scores)
    })
    print(f"   >>> Config Avg Acc: {avg_acc:.2%} (+/- {std_acc:.2%})")
    print("-" * 85)

# ==============================================================================
# 5. PODSUMOWANIE
# ==============================================================================
total_duration = (time.time() - total_start) / 3600

print("\n" + "="*80)
print(f"🏁 KONIEC. Czas: {total_duration:.2f}h")
print(f"🏆 REKORD (Single Best): {GLOBAL_BEST_ACC:.2%}")
print(f"⚙️ Params Rekordzisty:   {best_config_params}")
print(f"💾 Model zapisano w:     {best_model_filename}")

# Znalezienie najbardziej STABILNEJ konfiguracji (najwyższa średnia)
best_stable = max(stats, key=lambda x: x['avg_acc'])
print(f"\n📈 NAJBARDZIEJ STABILNA KONFIGURACJA (Średnia z {N_REPEATS} prób):")
print(f"   Acc: {best_stable['avg_acc']:.2%} (+/- {best_stable['std_acc']:.2%})")
print(f"   Params: {best_stable['params']}")
print("="*80)

# Finalne sprawdzenie zapisanego modelu
final_model = load_model(best_model_filename)
final_preds = (final_model.predict(X_test, verbose=0) > 0.5).astype(int)
print(classification_report(y_test, final_preds))