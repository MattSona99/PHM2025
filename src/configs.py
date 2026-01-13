# Per Grid Search, utilizzare SEMPRE liste, anche per un solo valore.
# Esempio: 'weight_decay': [1e-4, 1e-3, 1e-2] o 'weight_decay': [1e-4] se si vuole un solo valore.

CONFIGS = {
    
    # --- TRANSFORMER ---
    'transformer': {
        'window_size': [120],
        'learning_rate': [0.0001],
        'architecture': [(256, 4, 0.15)], # (Hidden, Heads, Dropout)
        'batch_size': [64],
        'epochs': [50],
        'num_layers': [3],
        'weight_decay': [1e-2],
        'clip_grad': [1.0],
        'pct_start': [0.3]
    },

    # --- xLSTM ---
    'xlstm': {
        'window_size': [120],
        'learning_rate': [0.001],
        'architecture': [(256, 4, 0.2)],
        'batch_size': [64],
        'epochs': [50],
        'num_layers': [2],
        'weight_decay': [1e-4],
        'clip_grad': [0.5],
        'pct_start': [0.2]
    },

    # --- DLinear ---
    'dlinear': {
        'window_size': [120],
        'learning_rate': [0.005],
        'architecture': [(0, 0, 0.1)],
        'batch_size': [64],
        'epochs': [50],
        'num_layers': [1],
        'weight_decay': [1e-4],
        'clip_grad': [5.0],
        'pct_start': [0.1]
    },


    # --- ENSEMBLE: Transformer + DLinear ---
    'dlinear_ensemble': {
        'window_size': [120],
        'learning_rate': [1e-4],
        'architecture': [(128, 4, 0.5)], 
        'batch_size': [128],
        'epochs': [60],
        'num_layers': [2],
        'weight_decay': [1e-2],
        'clip_grad': [1.0],
        'pct_start': [0.3]
    },
    
    # --- ENSEMBLE: xLSTM + Transformer ---
    'xlstm_ensemble': {
        'window_size': [50],
        'learning_rate': [0.0001],
        'architecture': [(128, 4, 0.5)],
        'batch_size': [128],
        'epochs': [50],
        'num_layers': [2],
        'weight_decay': [1e-3],
        'clip_grad': [1.0],
        'pct_start': [0.2]
    }
}

TARGET_COLS = ['Cycles_to_HPC_SV', 'Cycles_to_HPT_SV', 'Cycles_to_WW']