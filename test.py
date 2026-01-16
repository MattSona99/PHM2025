import torch
import pandas as pd
import glob
import joblib
import os
import inspect
import torch.utils.cpp_extension

# ==============================================================================
#  1. ZONA DI RIPARAZIONE CUDA PER WINDOWS
# ==============================================================================

# Configurazione dell'ambiente di compilazione JIT (Just-In-Time) per estensioni CUDA.
cuda_path = "Z:\\"
if not os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
    print("ERRORE CRITICO: Il drive Z: non è montato.")
    # Fallback o exit

# Configurazione delle variabili d'ambiente per il compilatore MSVC host (cl.exe).
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
    def apply_feature_engineering(df): return df

# =============================================================================
# 1. CONFIGURAZIONE DI LANCIO
# =============================================================================
MODEL_TO_TEST = 'dlinear_ensemble'

MODEL_MAPPING = {
    'transformer': Model_Transformer,
    'xlstm': Model_xLSTM,
    'dlinear': Model_DLinear,
    'dlinear_ensemble': Model_Ensemble_DLinear_Transformer,
    'xlstm_ensemble': Model_Ensemble_xLSTM_Transformer
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- INFERENCE DEVICE: {DEVICE} ---")

# =============================================================================
# 2. FUNZIONI DI UTILITÀ
# =============================================================================
def get_inference_config(model_type):
    if model_type not in CONFIGS:
        raise ValueError(f"Configurazione per '{model_type}' non trovata in src.configs")
    
    cfg = CONFIGS[model_type]
    base_dir = f"models/{model_type}/final/"
    
    return {
        'model_type': model_type,
        'window_size': cfg['window_size'][0],
        'architecture': cfg['architecture'][0],
        'num_layers': cfg.get('num_layers', [2])[0],
        'model_path': os.path.join(base_dir, "final_model.pth"),
        'pipeline_path': os.path.join(base_dir, "final_pipeline.pkl"),
        'output_file': f"data/results/{model_type}/submission.csv",
        'test_folder': 'data/raw/test/'
    }

def load_final_system(config):
    pipe_path = config['pipeline_path']
    if not os.path.exists(pipe_path):
        raise FileNotFoundError(f"Pipeline non trovata in: {pipe_path}. Hai eseguito train_full_model?")
    
    print(f"--- Caricamento Pipeline da: {pipe_path} ---")
    pipeline = joblib.load(pipe_path)
    
    model_type = config['model_type']
    model_path = config['model_path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello finale non trovato in: {model_path}")

    print(f"--- Caricamento Modello Finale da: {model_path} ---")
    ModelClass = MODEL_MAPPING.get(model_type)
    if not ModelClass: raise ValueError(f"Classe modello {model_type} non trovata.")
    
    hid, nhead, drop = config['architecture']
    
    all_params = {
        'n_features': pipeline.n_features,
        'n_engines': pipeline.n_engines,
        'n_profiles': pipeline.n_profiles,
        'n_targets': pipeline.n_targets,
        'emb_dim': 4,
        'hidden_size': hid,
        'nhead': nhead,
        'num_layers': config['num_layers'],
        'dropout': 0.0,
        'window_size': config['window_size']
    }
    
    sig = inspect.signature(ModelClass.__init__)
    valid_params = {k: v for k, v in all_params.items() if k in sig.parameters}
    
    model = ModelClass(**valid_params)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    return pipeline, model

# =============================================================================
# 3. LOOP DI INFERENZA
# =============================================================================
def process_batch():
    # A. Configurazione
    cfg = get_inference_config(MODEL_TO_TEST)
    print(f"--- CONFIGURAZIONE TEST FULL MODEL: {cfg['model_type'].upper()} ---")
    
    # B. Inizializzazione del sistema predittivo
    try:
        pipeline, model = load_final_system(cfg)
    except Exception as e:
        print(f"ERRORE INIZIALE: {e}")
        return

    # C. Scansione directory
    test_files = glob.glob(os.path.join(cfg['test_folder'], "*.csv"))
    print(f"--- Start Inferenza su {len(test_files)} files ---")
    
    all_results = []
    
    for file_path in test_files:
        # Estrae il nome del file e rimuove l'estensione (es. file_1.csv -> file_1)
        full_name = os.path.basename(file_path)
        file_name = os.path.splitext(full_name)[0]

        try:
            # 1. Ingestione dati
            df_raw = pd.read_csv(file_path)
            if 'Snapshot' not in df_raw.columns:
                 for col in ['cycle', 'Cycle', 'time', 'Time']:
                    if col in df_raw.columns:
                        df_raw.rename(columns={col: 'Snapshot'}, inplace=True); break
            
            # 2. Feature Engineering
            df_proc = apply_feature_engineering(df_raw)
            if df_proc.empty: continue

            # 3. Dummy Targets
            for col in TARGET_COLS:
                if col not in df_proc.columns: df_proc[col] = 0.0
            
            # 4. Pipeline Transform
            X_dict, _ = pipeline.transform(df_proc)
            if len(X_dict['input_sensors']) == 0: continue

            # 5. Tensori
            x_s = torch.tensor(X_dict['input_sensors'], dtype=torch.float32).to(DEVICE)
            x_e = torch.tensor(X_dict['input_esn'], dtype=torch.long).to(DEVICE)
            x_p = torch.tensor(X_dict['input_profile'], dtype=torch.long).to(DEVICE)
            
            # 6. Inferenza
            with torch.no_grad():
                out = model(x_s, x_e, x_p)
                pred_scaled = out[-1].cpu().numpy().reshape(1, -1)
            
            # Inversione scaling
            res_real = pipeline.target_scaler.inverse_transform(pred_scaled)[0]
            
            # 7. Risultati
            res_clamped = [max(1.0, val) for val in res_real]
            
            final_row = {
                'file': file_name,
                'Cycles_to_WW': res_clamped[2],
                'Cycles_to_HPC_SV': res_clamped[0],
                'Cycles_to_HPT_SV': res_clamped[1]
            }
            
            all_results.append(final_row)
            print(f"{file_name}: HPC={int(final_row['Cycles_to_HPC_SV'])} | HPT={int(final_row['Cycles_to_HPT_SV'])} | WW={int(final_row['Cycles_to_WW'])}")

        except Exception as e:
            print(f"Skipping {file_name}: {e}")
            import traceback
            traceback.print_exc()

# E. Esportazione risultati finali
    if all_results:
        os.makedirs(os.path.dirname(cfg['output_file']), exist_ok=True)
        df_res = pd.DataFrame(all_results).sort_values('file')
        
        # Riordino colonne per submission
        df_res = df_res[['file', 'Cycles_to_WW', 'Cycles_to_HPC_SV', 'Cycles_to_HPT_SV']]
        # ---------------------------------------
        
        df_res.to_csv(cfg['output_file'], index=False)
        print(f"\n--- SUCCESS ---")
        print(f"Submission generata in: {cfg['output_file']}")
        print("Colonne:", list(df_res.columns))
    else:
        print("\n--- NESSUN RISULTATO GENERATO ---")

if __name__ == "__main__":
    process_batch()