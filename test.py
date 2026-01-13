import torch
import pandas as pd
import numpy as np
import os
import glob
import inspect
import joblib
from src.preprocessing import Pipeline

# --- IMPORT CONFIGURAZIONI E MODELLI ---
from src.configs import CONFIGS, TARGET_COLS
from src.models import (
    Model_Transformer, 
    Model_xLSTM, 
    Model_DLinear,
    Model_Ensemble_DLinear_Transformer,
    Model_Ensemble_xLSTM_Transformer
)

# Gestione import feature engineering
try:
    from src.feature_engineering import apply_feature_engineering
except ImportError:
    # Fallback in caso di assenza del modulo specifico di ingegneria delle feature.
    # Si restituisce il dataframe grezzo per mantenere la continuità del flusso di inferenza.
    def apply_feature_engineering(df): return df

# =============================================================================
# 1. CONFIGURAZIONE DI LANCIO
# =============================================================================
# Selezione dell'architettura del modello da sottoporre a test di inferenza.
# Deve corrispondere all'architettura serializzata durante la fase di training completo (train_full_model).
# Modelli supportati: 'transformer', 'xlstm', 'dlinear', 'dlinear_ensemble', 'xlstm_ensemble'
MODEL_TO_TEST = 'transformer' 

# Mapping dei puntatori alle classi per l'istanziazione dinamica dei modelli predittivi.
MODEL_MAPPING = {
    'transformer': Model_Transformer,
    'xlstm': Model_xLSTM,
    'dlinear': Model_DLinear,
    'dlinear_ensemble': Model_Ensemble_DLinear_Transformer,
    'xlstm_ensemble': Model_Ensemble_xLSTM_Transformer
}

# Rilevamento automatico dell'acceleratore hardware disponibile per l'inferenza.
# L'utilizzo di CUDA è preferibile per ridurre la latenza di predizione su batch elevati.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- INFERENCE DEVICE: {DEVICE} ---")

# =============================================================================
# 2. FUNZIONI DI UTILITÀ
# =============================================================================
def get_inference_config(model_type):
    if model_type not in CONFIGS:
        raise ValueError(f"Configurazione per '{model_type}' non trovata in src.configs")
    
    cfg = CONFIGS[model_type]
    
    # Definizione del percorso base per il recupero degli artefatti del modello di produzione (Full Dataset).
    base_dir = f"models/{model_type}/final/"
    
    # Costruzione del dizionario di configurazione per l'ambiente di test.
    # Si estraggono i parametri architetturali (finestra temporale, layer, hidden size)
    # necessari per ricostruire il grafo computazionale identico a quello di training.
    return {
        'model_type': model_type,
        'window_size': cfg['window_size'][0],
        'architecture': cfg['architecture'][0], # (Hidden, Heads, Dropout)
        'num_layers': cfg.get('num_layers', [2])[0],
        
        # Percorsi specifici del FULL MODEL
        'model_path': os.path.join(base_dir, "final_model.pth"),
        'pipeline_path': os.path.join(base_dir, "final_pipeline.pkl"),
        
        'output_file': f"data/results/{model_type}/submission_final.csv",
        'test_folder': 'data/raw/test/'
    }

def load_final_system(config):
    
    # 1. Deserializzazione della Pipeline di Preprocessing.
    # Fondamentale per garantire che i dati di test subiscano la stessa normalizzazione (Scaling)
    # e trasformazione dei dati di training.
    pipe_path = config['pipeline_path']
    if not os.path.exists(pipe_path):
        raise FileNotFoundError(f"Pipeline non trovata in: {pipe_path}. Hai eseguito train_full_model?")
    
    print(f"--- Caricamento Pipeline da: {pipe_path} ---")
    pipeline = joblib.load(pipe_path)
    
    # 2. Ricostruzione dell'architettura del modello neurale.
    model_type = config['model_type']
    model_path = config['model_path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello finale non trovato in: {model_path}")

    print(f"--- Caricamento Modello Finale da: {model_path} ---")
    ModelClass = MODEL_MAPPING.get(model_type)
    if not ModelClass: raise ValueError(f"Classe modello {model_type} non trovata.")
    
    hid, nhead, drop = config['architecture']
    
    # Definizione parametri costruttivi. 
    # Il dropout viene forzato a 0.0 poiché in fase di inferenza si richiede un comportamento deterministico.
    all_params = {
        'n_features': pipeline.n_features,
        'n_engines': pipeline.n_engines,
        'n_profiles': pipeline.n_profiles,
        'n_targets': pipeline.n_targets,
        'emb_dim': 4,
        'hidden_size': hid,
        'nhead': nhead,
        'num_layers': config['num_layers'],
        'dropout': 0.0, # Dropout spento in inferenza
        'window_size': config['window_size']
    }
    
    # Validazione dei parametri rispetto alla firma del costruttore tramite introspezione.
    sig = inspect.signature(ModelClass.__init__)
    valid_params = {k: v for k, v in all_params.items() if k in sig.parameters}
    
    # Istanziazione del modello
    model = ModelClass(**valid_params)
    
    # Caricamento dei pesi (state_dict) addestrati e trasferimento su device.
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    
    # Impostazione della modalità di valutazione (disattiva BatchNorm update e Dropout).
    model.eval()
    
    return pipeline, model

# =============================================================================
# 3. LOOP DI INFERENZA
# =============================================================================
def process_batch():
    # A. Configurazione
    cfg = get_inference_config(MODEL_TO_TEST)
    print(f"--- CONFIGURAZIONE TEST FULL MODEL: {cfg['model_type'].upper()} ---")
    
    # B. Inizializzazione del sistema predittivo (Pipeline + Rete Neurale)
    try:
        pipeline, model = load_final_system(cfg)
    except Exception as e:
        print(f"ERRORE INIZIALE: {e}")
        return

    # C. Scansione directory dei file di telemetria per inferenza batch.
    test_files = glob.glob(os.path.join(cfg['test_folder'], "*.csv"))
    print(f"--- Start Inferenza su {len(test_files)} files ---")
    
    all_results = []
    
    for file_path in test_files:
        file_name = os.path.basename(file_path)
        try:
            # 1. Ingestione dati grezzi e normalizzazione nomenclatura colonne temporali.
            df_raw = pd.read_csv(file_path)
            if 'Snapshot' not in df_raw.columns:
                 for col in ['cycle', 'Cycle', 'time', 'Time']:
                    if col in df_raw.columns:
                        df_raw.rename(columns={col: 'Snapshot'}, inplace=True); break
            
            # 2. Applicazione logiche di Feature Engineering specifiche per il dominio.
            df_proc = apply_feature_engineering(df_raw)
            if df_proc.empty: continue

            # 3. Creazione target fittizi (Dummy Targets).
            # Necessari per soddisfare l'interfaccia della pipeline di trasformazione che si aspetta colonne target,
            # anche se in fase di test il ground truth non è noto.
            for col in TARGET_COLS:
                if col not in df_proc.columns: df_proc[col] = 0.0
            
            # 4. Applicazione della trasformazione (Scaling/Encoding) tramite Pipeline serializzata.
            X_dict, _ = pipeline.transform(df_proc)
            if len(X_dict['input_sensors']) == 0: continue

            # 5. Conversione in tensori PyTorch e trasferimento su GPU/CPU.
            x_s = torch.tensor(X_dict['input_sensors'], dtype=torch.float32).to(DEVICE)
            x_e = torch.tensor(X_dict['input_esn'], dtype=torch.long).to(DEVICE)
            x_p = torch.tensor(X_dict['input_profile'], dtype=torch.long).to(DEVICE)
            
            # 6. Esecuzione inferenza del modello (Forward Pass).
            with torch.no_grad():
                out = model(x_s, x_e, x_p)
                # Si estrae l'output corrispondente all'ultimo step temporale della sequenza
                # per ottenere la stima RUL aggiornata all'ultimo ciclo disponibile.
                pred_scaled = out[-1].cpu().numpy().reshape(1, -1)
            
            # Inversione dello scaling per ottenere valori RUL in cicli fisici reali.
            res_real = pipeline.target_scaler.inverse_transform(pred_scaled)[0]
            
            # 7. Post-processing e raccolta risultati.
            # Clipping inferiore a 1.0 per evitare valori RUL negativi o nulli non fisici.
            res_clamped = [max(1.0, val) for val in res_real]
            
            final_row = {
                'File': file_name,
                'Pred_HPC_SV': res_clamped[0],
                'Pred_HPT_SV': res_clamped[1],
                'Pred_WW':     res_clamped[2]
            }
            
            all_results.append(final_row)
            print(f"{file_name}: HPC={int(final_row['Pred_HPC_SV'])} | HPT={int(final_row['Pred_HPT_SV'])} | WW={int(final_row['Pred_WW'])}")

        except Exception as e:
            print(f"Skipping {file_name}: {e}")
            import traceback
            traceback.print_exc()

    # E. Esportazione risultati finali
    if all_results:
        os.makedirs(os.path.dirname(cfg['output_file']), exist_ok=True)
        df_res = pd.DataFrame(all_results).sort_values('File')
        
        # Selezione e ordinamento colonne per conformità al formato di sottomissione.
        df_res = df_res[['File', 'Pred_HPC_SV', 'Pred_HPT_SV', 'Pred_WW']]
        
        df_res.to_csv(cfg['output_file'], index=False)
        print(f"\n--- SUCCESS ---")
        print(f"Submission generata in: {cfg['output_file']}")
    else:
        print("\n--- NESSUN RISULTATO GENERATO ---")

if __name__ == "__main__":
    process_batch()