import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import itertools
import inspect
import time
import torch.utils.cpp_extension

# ==============================================================================
#  1. ZONA DI RIPARAZIONE CUDA PER WINDOWS
# ==============================================================================

# Configurazione dell'ambiente di compilazione JIT (Just-In-Time) per estensioni CUDA.
# Si rende necessaria la mappatura su drive virtuale Z: per aggirare le limitazioni 
# sulla lunghezza dei path in ambiente Windows durante il linking con NVCC.
cuda_path = "Z:\\"
if not os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
    print("ERRORE CRITICO: Il drive Z: non è montato.")
    # Fallback o exit

# Configurazione delle variabili d'ambiente per il compilatore MSVC host (cl.exe).
# Si definiscono i flag di inclusione header e linking librerie per garantire 
# la compatibilità tra torch.utils.cpp_extension e il toolkit CUDA installato.
path_to_cl = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
os.environ["CUDA_HOME"] = cuda_path
os.environ["CUDA_PATH"] = cuda_path
os.environ["CC"] = path_to_cl
os.environ["CXX"] = path_to_cl
os.environ["MAX_JOBS"] = "1"
os.environ["PATH"] = (
        os.path.join(cuda_path, "bin") + ";" +
        os.path.join(cuda_path, "include") + ";" +
        os.path.join(cuda_path, "lib", "x64") + ";" +
        os.environ["PATH"]
)
os.environ["CFLAGS"] = f"-I{cuda_path}include"
os.environ["CXXFLAGS"] = f"-I{cuda_path}include"
os.environ["DISTUTILS_USE_SDK"] = "1"

original_load = torch.utils.cpp_extension.load

# Override della funzione di caricamento estensioni per iniettare flag di compatibilità.
# Vengono filtrati flag di ottimizzazione non supportati (es. -O3 su nvcc specifici)
# e aggiunte definizioni per la gestione delle discrepanze di versione STL.
def patched_load(*args, **kwargs):
    if 'extra_cuda_cflags' in kwargs and kwargs['extra_cuda_cflags']:
        new_flags = []
        new_flags.append('-allow-unsupported-compiler')
        new_flags.append('-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH')

        for flag in kwargs['extra_cuda_cflags']:
            if '-Xptxas' in flag and '-O3' in flag:
                new_flags.append('-Xptxas')
                new_flags.append('-O3')
            elif flag.strip() == '-O3':
                new_flags.append('-O3')
            else:
                new_flags.append(flag)
        kwargs['extra_cuda_cflags'] = new_flags

    if 'extra_ldflags' in kwargs and kwargs['extra_ldflags']:
        new_ldflags = []
        for flag in kwargs['extra_ldflags']:
            if flag.startswith('-L'):
                new_ldflags.append(f"/LIBPATH:{cuda_path}lib\\x64")
            elif flag.startswith('-l'):
                lib_name = flag[2:]
                if lib_name == 'cublas':
                    new_ldflags.append('cublas.lib')
                elif lib_name == 'cudart':
                    new_ldflags.append('cudart.lib')
                else:
                    new_ldflags.append(f"{lib_name}.lib")
            else:
                new_ldflags.append(flag)
        kwargs['extra_ldflags'] = new_ldflags

    return original_load(*args, **kwargs)


torch.utils.cpp_extension.load = patched_load
print("--- AMBIENTE CORRETTO E PATCH APPLICATA ---")

# =============================================================================
# 1. SETUP E IMPORT
# =============================================================================
from src.preprocessing import Pipeline
from src.configs import CONFIGS, TARGET_COLS
from src.models import (
    Model_Transformer, 
    Model_xLSTM,
    Model_DLinear,
    Model_Ensemble_DLinear_Transformer,
    Model_Ensemble_xLSTM_Transformer
)

print("--- SETUP AMBIENTE ---")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizzato: {DEVICE}")

# Mapping delle architetture di rete neurale disponibili per la stima del RUL (Remaining Useful Life).
MODEL_MAPPING = {
    'transformer': Model_Transformer,
    'xlstm': Model_xLSTM,
    'dlinear': Model_DLinear,
    'dlinear_ensemble': Model_Ensemble_DLinear_Transformer,
    'xlstm_ensemble': Model_Ensemble_xLSTM_Transformer
}

# Definizione del Dataset Custom per l'ingestione dei dati di telemetria.
# Si strutturano i tensori separando i dati dei sensori (serie temporali continue),
# gli identificativi dell'asset (ESN) e i profili operativi.
class PHMDataset(Dataset):
    def __init__(self, X_dict, y):
        self.x_sens = torch.tensor(X_dict['input_sensors'], dtype=torch.float32)
        self.x_esn = torch.tensor(X_dict['input_esn'], dtype=torch.long)
        self.x_prof = torch.tensor(X_dict['input_profile'], dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return (self.x_sens[idx], self.x_esn[idx], self.x_prof[idx]), self.y[idx]

# =============================================================================
# 2. METRICHE
# =============================================================================
# Calcolo della metrica di competizione.
# Si implementa una funzione di errore pesata nel tempo che penalizza maggiormente 
# gli errori commessi quando il componente è prossimo al fine vita, favorendo 
# previsioni conservative (late prediction penalty).
def calculate_competition_score(y_true, y_pred):
    def time_weighted_error(y_t, y_p, beta):
        err = y_p - y_t
        weight = np.where(err >= 0, 2 / (1 + 0.01 * y_t), 1 / (1 + 0.01 * y_t))
        return weight * (err ** 2) * beta

    s_hpc = np.mean(time_weighted_error(y_true[:,0], y_pred[:,0], beta=2/np.max(y_true[:,0])))
    s_hpt = np.mean(time_weighted_error(y_true[:,1], y_pred[:,1], beta=2/np.max(y_true[:,1])))
    s_ww = np.mean(time_weighted_error(y_true[:,2], y_pred[:,2], beta=1/np.max(y_true[:,2])))
    
    return np.mean([s_ww, s_hpc, s_hpt])

# Implementazione di una Loss Function asimmetrica per la regressione del RUL.
# Si applica una penalità lineare/quadratica (Huber) modulata da un fattore asimmetrico:
# le stime sovrastimate (RUL predetto > RUL reale) comportano un rischio operativo maggiore
# e vengono penalizzate con un fattore 'late_penalty'.
class AsymmetricHuberLoss(nn.Module):
    def __init__(self, delta=1.0, late_penalty=2.0):
        super().__init__()
        self.delta = delta
        self.late_penalty = late_penalty

    def forward(self, pred, target):
        error = pred - target

        # Calcolo Huber Loss standard
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        base_loss = 0.5 * quadratic ** 2 + self.delta * linear

        # Creazione peso asimmetrico
        # Se error > 0 (predizione in ritardo/pericolosa), si moltiplica per late_penalty
        # Se error <= 0 (predizione in anticipo/sicura), si moltiplica per 1.0
        asymmetric_weight = torch.where(error > 0, self.late_penalty, 1.0)

        # Loss pesata
        return torch.mean(asymmetric_weight * base_loss)
    
# =============================================================================
# 3. TRAINING LOOP UNIFICATO
# =============================================================================
def train_single_fold(config, df_train, df_val, fold_idx):
    model_type = config['model_type']
    
    # Setup Cartelle per persistenza modelli e log metriche
    model_save_dir = f"models/{model_type}/"
    result_save_dir = f"data/results/{model_type}/"
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)

    # Inizializzazione Pipeline di Preprocessing e Scaling
    pipeline = Pipeline(windows_size=config['window_size'], target_cols=TARGET_COLS)
    pipeline.fit(df_train)
    
    X_tr, y_tr = pipeline.transform(df_train)
    X_val, y_val = pipeline.transform(df_val)
    
    train_loader = DataLoader(PHMDataset(X_tr, y_tr), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(PHMDataset(X_val, y_val), batch_size=config['batch_size'], shuffle=False)
    
    # Istanziazione Modello tramite Reflection sui parametri del costruttore
    ModelClass = MODEL_MAPPING.get(model_type)
    if not ModelClass: raise ValueError(f"Modello {model_type} non trovato.")
    
    hid, nhead, drop = config['architecture']
    
    # Configurazione dinamica iperparametri architetturali
    all_params = {
        'n_features': pipeline.n_features,
        'n_engines': pipeline.n_engines,
        'n_profiles': pipeline.n_profiles,
        'n_targets': pipeline.n_targets,
        'emb_dim': 4,
        'hidden_size': hid,
        'nhead': nhead,
        'num_layers': config.get('num_layers', 2),
        'dropout': drop,
        'window_size': config['window_size']
    }
    
    # Filtro argomenti validi in base alla firma della classe modello
    sig = inspect.signature(ModelClass.__init__)
    valid_params = {k: v for k, v in all_params.items() if k in sig.parameters}
    
    model = ModelClass(**valid_params).to(DEVICE)
    
    # Compilazione JIT del modello per ottimizzazioni runtime (se supportato)
    if sys.platform != "win32":
        model = torch.compile(model)
    
    # --- CONFIGURAZIONE OTTIMIZZATORE E REGOLARIZZAZIONE ---
    # Si recuperano i parametri di weight decay e gradient clipping per il controllo
    # della convergenza e la prevenzione dell'overfitting sui dati di sensore rumorosi.
    weight_decay = config.get('weight_decay', 1e-4)
    clip_grad = config.get('clip_grad', 1.0)
    pct_start = config.get('pct_start', 0.3)
    
    criterion = AsymmetricHuberLoss(delta=0.1, late_penalty=2.0)
    
    # 1. Optimizer AdamW con disaccoppiamento del weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=weight_decay)
    
    # 2. Scheduler OneCycleLR per warm-up e annealing del learning rate
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'], 
                                              epochs=config['epochs'], steps_per_epoch=len(train_loader),
                                              pct_start=pct_start)
    
    best_score = float('inf')
    counter = 0
    patience = 15
    train_hist, val_hist = [], []
    
    print(f"   [Fold {fold_idx}] Start Training ({config['epochs']} epochs) | WD: {weight_decay} | Clip: {clip_grad}")
    
    for epoch in range(config['epochs']):
        model.train()
        avg_loss = 0
        for inp, tgt in train_loader:
            x_s, x_e, x_p = [x.to(DEVICE) for x in inp]
            tgt = tgt.to(DEVICE)
            optimizer.zero_grad()
            out = model(x_s, x_e, x_p)
            loss = criterion(out, tgt)
            loss.backward()
            
            # 3. Applicazione del Gradient Clipping per stabilità numerica
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item()
        train_hist.append(avg_loss / len(train_loader))
        
        # Validazione e Calcolo Score Competizione
        model.eval()
        preds, targets, v_loss_accum = [], [], 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                x_s, x_e, x_p = [x.to(DEVICE) for x in inp]
                tgt = tgt.to(DEVICE)
                out = model(x_s, x_e, x_p)
                v_loss_accum += criterion(out, tgt).item()
                preds.append(out.cpu().numpy())
                targets.append(tgt.cpu().numpy())
        val_hist.append(v_loss_accum / len(val_loader))
        
        # Inversione scaling per valutazione su scale fisiche reali
        y_p_inv = pipeline.target_scaler.inverse_transform(np.concatenate(preds))
        y_t_inv = pipeline.target_scaler.inverse_transform(np.concatenate(targets))
        score = calculate_competition_score(y_t_inv, y_p_inv)
        
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), f"{model_save_dir}model_fold{fold_idx}.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience: break
            
    # Salvataggio curve di apprendimento per diagnostica addestramento
    plt.figure(figsize=(10,6))
    plt.plot(train_hist, label='Train Loss'); plt.plot(val_hist, label='Val Loss')
    plt.title(f"{model_type.upper()} Fold {fold_idx} | Best: {best_score:.2f}")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{result_save_dir}loss_fold{fold_idx}.png")
    plt.close()
    
    return best_score

# =============================================================================
# 4. CROSS VALIDATION RUNNER
# =============================================================================
def run_cv(config, df):
    model_type = config['model_type']
    print(f"\n>>> ARCHITETTURA: {model_type.upper()}")
    print(f"    Params: Win={config['window_size']}, LR={config['learning_rate']}, Arch={config['architecture']}")
    
    # --- Inizio monitoraggio tempi esecuzione ---
    start_time = time.time()
    
    # Strategia di validazione: Leave-One-Group-Out basata su Engine Serial Number (ESN).
    # Si valuta la capacità di generalizzazione del modello su asset fisici non visti in training.
    engines = df['ESN'].unique()
    scores = []
    
    for i, eng in enumerate(engines):
        print(f" -> Val Engine: {eng}...", end="")
        tr = df[df['ESN'] != eng].copy()
        val = df[df['ESN'] == eng].copy()
        s = train_single_fold(config, tr, val, i+1)
        scores.append(s)
        print(f" Best Score: {s:.4f}")
    
    # --- Fine monitoraggio ---
    end_time = time.time()
    elapsed_time = end_time - start_time
        
    avg, std = np.mean(scores), np.std(scores)
    print(f" >>> {model_type.upper()} RESULT: {avg:.4f} (+/- {std:.2f}) | Time: {elapsed_time:.2f}s")
    
    return avg, std, scores, elapsed_time

import joblib

# Funzione per l'addestramento del modello finale di produzione.
# Viene utilizzato l'intero dataset disponibile per massimizzare l'informazione appresa
# prima del deployment o della fase di inferenza.
def train_full_model(config, df_all):
    print(f"\n>>> AVVIO RETRAINING SU TUTTO IL DATASET (NO VALIDATION SPLIT)")
    
    model_type = config['model_type']
    save_dir = f"models/{model_type}/final/"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Fitting della pipeline di preprocessing sull'intera popolazione statistica
    pipeline = Pipeline(windows_size=config['window_size'], target_cols=TARGET_COLS)
    pipeline.fit(df_all)
    
    joblib.dump(pipeline, f"{save_dir}final_pipeline.pkl")
    print("Pipeline scalers salvati.")

    # Trasformazione dati
    X_all, y_all = pipeline.transform(df_all)
    
    # Creazione Dataset e Loader unico per training batch
    full_dataset = PHMDataset(X_all, y_all)
    train_loader = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=True)

    # 2. Istanziazione Modello Finale
    ModelClass = MODEL_MAPPING.get(model_type)
    hid, nhead, drop = config['architecture']
    
    all_params = {
        'n_features': pipeline.n_features,
        'n_engines': pipeline.n_engines,
        'n_profiles': pipeline.n_profiles,
        'n_targets': pipeline.n_targets,
        'emb_dim': 4,
        'hidden_size': hid,
        'nhead': nhead,
        'num_layers': config.get('num_layers', 2),
        'dropout': drop,
        'window_size': config['window_size']
    }
    
    sig = inspect.signature(ModelClass.__init__)
    valid_params = {k: v for k, v in all_params.items() if k in sig.parameters}
    
    model = ModelClass(**valid_params).to(DEVICE)
    
    # 3. Setup Parametri di Training (replicando la config migliore)
    weight_decay = config.get('weight_decay', 1e-4)
    clip_grad = config.get('clip_grad', 1.0)
    pct_start = config.get('pct_start', 0.3)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'], 
                                              epochs=config['epochs'], steps_per_epoch=len(train_loader),
                                              pct_start=pct_start)
    criterion = AsymmetricHuberLoss(delta=0.1, late_penalty=2.0)

    # 4. Training Loop (senza fase di validazione)
    print("Inizio addestramento finale...")
    model.train()
    
    for epoch in range(config['epochs']):
        avg_loss = 0
        for inp, tgt in train_loader:
            x_s, x_e, x_p = [x.to(DEVICE) for x in inp]
            tgt = tgt.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(x_s, x_e, x_p)
            loss = criterion(out, tgt)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss/len(train_loader):.6f}")

    # 5. Serializzazione dei pesi del modello addestrato
    final_path = f"{save_dir}final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"MODELLO FINALE SALVATO IN: {final_path}")
    print("==================================================")

# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    data_path = 'data/processed/final_training_data.csv'
    if not os.path.exists(data_path): print("Dataset non trovato"); return
    df = pd.read_csv(data_path)
    
    # --- SELEZIONE MODELLI DA ESEGUIRE ---
    # Selezione dell'architettura target per la campagna sperimentale.
    # Modelli supportati: 'transformer', 'xlstm', 'dlinear', 'dlinear_ensemble', 'xlstm_ensemble'
    MODELS_TO_RUN = ['dlinear_ensemble']
    
    experiment_counter = 1
    
    for model_name in MODELS_TO_RUN:
        if model_name not in CONFIGS:
            print(f"ATTENZIONE: Configurazione per '{model_name}' non trovata in src/configs.py. Salto.")
            continue
            
        # Caricamento griglia iperparametri dal file di configurazione
        param_grid = CONFIGS[model_name]
        param_grid['model_type'] = [model_name]
        
        # Generazione combinazioni per Grid Search
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"\n==================================================")
        print(f"AVVIO TRAINING PER: {model_name.upper()}")
        print(f"Configurazioni totali: {len(combinations)}")
        print(f"==================================================")
        
        best_avg_score = float('inf')
        best_config = None
        results = []

        # Iterazione sulle configurazioni sperimentali
        for cfg in combinations:
            try:
                avg, std, scores, elapsed_time = run_cv(cfg, df)
                
                # Logging strutturato dei risultati
                entry = cfg.copy()
                entry['avg_score'] = avg
                entry['std_score'] = std
                entry['fold_scores'] = str(scores)
                entry['execution_time'] = elapsed_time
                
                results.append(entry)
                
                # Aggiornamento del miglior modello (criterio: minimo score pesato)
                if avg < best_avg_score:
                    best_avg_score = avg
                    best_config = cfg
                    print(f"   >>> Nuova miglior configurazione trovata! (Score: {best_avg_score:.4f})")
                    
                # Salvataggio incrementale su CSV per tracciabilità esperimenti
                res_path = f"data/results/{model_name}/training_summary.csv"
                os.makedirs(os.path.dirname(res_path), exist_ok=True)
                pd.DataFrame(results).to_csv(res_path, index=False)
                
                experiment_counter += 1
                
            except Exception as e:
                print(f"ERRORE CRITICO DURANTE CV: {e}")
                import traceback; traceback.print_exc()

        # Retraining finale utilizzando la configurazione ottimale identificata
        if best_config is not None:
            print(f"\n>>> Miglior Config trovata (Score: {best_avg_score:.4f}).")
            print(">>> Avvio Retraining sul dataset completo...")
            try:
                train_full_model(best_config, df)      
            except Exception as e:
                print(f"ERRORE CRITICO DURANTE FULL RETRAINING: {e}")
                import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()