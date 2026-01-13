import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor

# ==========================================
# 1. SEGMENTAZIONE FASI DI VOLO
# ==========================================
def segment_flight_phases(df):
    """
    Algoritmo di riconoscimento del regime operativo (Flight Regime Recognition).
    Si segmenta la serie temporale del volo identificando le transizioni di stato
    tramite l'analisi del gradiente altimetrico (Rate of Climb) e della velocità (Mach).
    """
    df_out = df.copy()
    cycle_col = 'Cycles_Since_New' if 'Cycles_Since_New' in df_out.columns else 'Cycles'
    
    # Ordinamento cronologico per garantire la coerenza delle derivate temporali.
    if 'Snapshot' in df_out.columns:
        sort_cols = ['ESN', cycle_col] if cycle_col in df_out.columns else ['ESN']
        df_out = df_out.sort_values(sort_cols + ['Snapshot'])

    # Gestione fallback per dati incompleti: assunzione di regime stazionario (CRUISE).
    if 'Sensed_Altitude' not in df_out.columns or 'Sensed_Mach' not in df_out.columns:
        df_out['Flight_Phase'] = 'CRUISE'
        return df_out

    # Calcolo dei differenziali cinematici per singolo ciclo motore.
    grp = df_out.groupby(['ESN', cycle_col]) if cycle_col in df_out.columns else df_out.groupby('ESN')
    df_out['Alt_Delta'] = grp['Sensed_Altitude'].diff().fillna(0)
    df_out['Alt_Delta_Next'] = (grp['Sensed_Altitude'].shift(-1) - df_out['Sensed_Altitude']).fillna(0)

    # Normalizzazione dell'asse temporale del volo (Flight Progress [0,1]).
    max_snap = grp['Snapshot'].transform('max')
    min_snap = grp['Snapshot'].transform('min')
    denom = max_snap - min_snap
    df_out['Flight_Progress'] = np.where(denom <= 0, 0.5, (df_out['Snapshot'] - min_snap) / denom)

    # Definizione della soglia di riferimento terra (Ground Level) dinamica per asset.
    df_out['Alt_Ground'] = grp['Sensed_Altitude'].transform('min')
    df_out['Ground_Threshold'] = df_out['Alt_Ground'] + 1500 

    # Parametrizzazione delle soglie cinematiche per la macchina a stati finiti.
    THRESH_MACH_MOVE = 0.15
    THRESH_RATE_CLIMB = 300  
    THRESH_RATE_DESC = -300

    is_ground = df_out['Sensed_Altitude'] < df_out['Ground_Threshold']
    is_climb = df_out['Alt_Delta'] > THRESH_RATE_CLIMB
    is_desc = df_out['Alt_Delta'] < THRESH_RATE_DESC

    # Logica decisionale per l'assegnazione della fase di volo.
    # 
    conditions = [
        (is_ground) & (df_out['Sensed_Mach'] < THRESH_MACH_MOVE) & (df_out['Flight_Progress'] < 0.3),
        (is_ground) & (df_out['Sensed_Mach'] < THRESH_MACH_MOVE) & (df_out['Flight_Progress'] > 0.7),
        (is_ground) & ((is_climb) | (df_out['Alt_Delta_Next'] > THRESH_RATE_CLIMB)),                  
        (is_ground) & (is_desc) & (df_out['Flight_Progress'] > 0.6),                                  
        (~is_ground) & (is_climb),                                                                     
        (~is_ground) & (is_desc),                                                                      
        (~is_ground) & (df_out['Alt_Delta'].abs() <= 200)                                              
    ]
    choices = ['TAXI-OUT', 'TAXI-IN', 'TAKEOFF', 'LANDING', 'CLIMB', 'DESCENT', 'CRUISE']
    df_out['Flight_Phase'] = np.select(conditions, choices, default='UNKNOWN')
    
    # Rimozione variabili temporanee di calcolo cinematico.
    cols_to_drop = ['Alt_Delta', 'Alt_Delta_Next', 'Flight_Progress', 'Alt_Ground', 'Ground_Threshold']
    df_out.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df_out

# ==========================================
# 2. OPERATIONAL PROFILE
# ==========================================
def detect_operational_profile(df):
    """
    Caratterizzazione del profilo di missione (Severity Analysis).
    Si discrimina l'utilizzo del motore analizzando la velocità del core
    durante la fase di salita (CLIMB), identificando eventuali derating 
    della spinta o profili ad alto carico termico.
    """
    df['Operational_Profile'] = 0
    if 'Sensed_Core_Speed' in df.columns and 'Flight_Phase' in df.columns:
        SPEED_THRESHOLD = 20200
        climb_data = df[df['Flight_Phase'] == 'CLIMB']
        if not climb_data.empty:
            if 'ESN' in df.columns:
                avg_speed = climb_data.groupby('ESN')['Sensed_Core_Speed'].mean()
                # Identificazione motori operanti a regime ridotto (Low Severity)
                engines_reduced = avg_speed[avg_speed < SPEED_THRESHOLD].index.tolist()
                mask = (df['ESN'].isin(engines_reduced))
                df.loc[mask, 'Operational_Profile'] = 1
            else:
                if climb_data['Sensed_Core_Speed'].mean() < SPEED_THRESHOLD:
                    df['Operational_Profile'] = 1
    return df

# ==========================================
# 3. CALCOLO RESIDUI
# ==========================================
def calculate_physics_residuals(df):
    """
    Calcolo dei residui basati su modello fisico (Physics-Based Residuals).
    Si stabilisce una baseline termodinamica utilizzando i primi cicli di vita (Healthy State)
    per predire i parametri attesi (Target) in funzione delle condizioni operative (Features).
    La deviazione (Residuo) rappresenta l'indicatore di salute (HI) depurato dalle variazioni ambientali.
    """
    
    # Definizione delle relazioni termodinamiche tra stazioni motore monitorate.
    # 
    configs = [
        # Relazione Compressore Alta Pressione (HPC): P25, N2 -> Ps3
        {'name': 'HPC_Resid', 'features': ['Sensed_P25', 'Sensed_Core_Speed'], 'target': 'Sensed_Ps3', 'phases': ['CLIMB', 'TAKEOFF']},
        # Relazione Turbina Alta Pressione (HPT): N2, P25 -> T45 (EGT)
        {'name': 'HPT_Resid', 'features': ['Sensed_Core_Speed', 'Sensed_P25'], 'target': 'Sensed_T45', 'phases': ['CLIMB', 'TAKEOFF']},
        # Relazione Globale crociera: T3 -> T45
        {'name': 'Global_Resid', 'features': ['Sensed_T3'], 'target': 'Sensed_T45', 'phases': ['CRUISE']}
    ]
    
    # Inizializzazione e gestione robusta feature mancanti (Zero Imputation).
    required_cols = set()
    for cfg in configs:
        required_cols.update(cfg['features'])
        required_cols.add(cfg['target'])
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    for cfg in configs:
        if cfg['name'] not in df.columns: df[cfg['name']] = 0.0

    cycle_col = 'Cycles_Since_New' if 'Cycles_Since_New' in df.columns else 'Cycles'
    if 'ESN' not in df.columns: df['ESN'] = 0
    
    # --- TRAINING BASELINE FLOTTA ---
    # Addestramento di modelli di regressione robusta (Huber) sui dati di flotta
    # relativi ai primi 50 cicli, per catturare il comportamento nominale medio.
    fleet_models = {}
    for cfg in configs:
        mask_fleet = (df[cycle_col] <= 50) & (df['Flight_Phase'].isin(cfg['phases']))
        df_fleet = df[mask_fleet].dropna(subset=cfg['features'] + [cfg['target']])
        
        if len(df_fleet) > 50: 
            try:
                # HuberRegressor è scelto per la sua resistenza agli outlier nei dati di sensore.
                model = HuberRegressor(epsilon=1.35)
                model.fit(df_fleet[cfg['features']], df_fleet[cfg['target']])
                fleet_models[cfg['name']] = model
            except:
                fleet_models[cfg['name']] = None
        else:
            fleet_models[cfg['name']] = None

    # --- CALCOLO RESIDUI ASSET-SPECIFIC ---
    # Per ogni motore, si tenta di affinare la baseline sui propri dati iniziali.
    # Se i dati storici dell'asset sono insufficienti, si ricorre al modello di flotta (Fallback).
    unique_engines = df['ESN'].unique()
    for esn in unique_engines:
        for cfg in configs:
            feats = cfg['features']
            target = cfg['target']
            
            mask_valid = (df['ESN'] == esn) & (df['Flight_Phase'].isin(cfg['phases']))
            df_subset = df[mask_valid]
            if df_subset.empty: continue

            min_cycle = df_subset[cycle_col].min()
            model = None
            
            # Calibrazione locale se l'asset è osservato fin dall'inizio vita (BOL - Beginning of Life).
            if min_cycle <= 50:
                train_mask = df_subset[cycle_col] <= (min_cycle + 50)
                df_train = df_subset[train_mask].dropna(subset=feats + [target])
                if len(df_train) > 10:
                    try:
                        model = LinearRegression()
                        model.fit(df_train[feats], df_train[target])
                    except: pass
            
            if model is None:
                model = fleet_models[cfg['name']]

            # Calcolo e salvataggio del residuo (Actual - Expected).
            if model is not None:
                X_all = df_subset[feats].ffill().fillna(0)
                expected = model.predict(X_all)
                df.loc[mask_valid, cfg['name']] = df_subset[target] - expected

    return df

# ==========================================
# 4. SMOOTHING
# ==========================================
def apply_smoothing(df, window=10):
    """
    Filtraggio del segnale (Signal Denoising).
    Applicazione di una media mobile (Rolling Mean) per abbattere il rumore di misura
    ad alta frequenza e far emergere il trend di degradazione sottostante.
    """
    cols_to_smooth = ['HPC_Resid', 'HPT_Resid', 'Global_Resid']
    cols_present = [c for c in cols_to_smooth if c in df.columns]
    if not cols_present: return df

    cycle_col = 'Cycles_Since_New' if 'Cycles_Since_New' in df.columns else 'Cycles'
    # Aggregazione per ciclo per gestire la variabilità intra-volo
    df_cycle = df.groupby(['ESN', cycle_col])[cols_present].mean().reset_index()
    frames = []

    for esn, group in df_cycle.groupby('ESN'):
        group = group.sort_values(cycle_col).copy()
        for col in cols_present:
            new_col = f"{col}_Smooth"
            group[new_col] = group[col].rolling(window=window, min_periods=1, center=False).mean()
        frames.append(group)

    if not frames: return df
    
    # Join dei segnali filtrati sul dataframe originale ad alta risoluzione.
    df_smooth = pd.concat(frames)
    cols_smooth_names = [f"{c}_Smooth" for c in cols_present]
    df_to_merge = df_smooth[['ESN', cycle_col] + cols_smooth_names]
    
    df_out = df.merge(df_to_merge, on=['ESN', cycle_col], how='left')
    
    # Interpolazione (Forward Fill) per mantenere la continuità del segnale.
    for col in cols_smooth_names:
        if 'ESN' in df_out.columns:
            df_out[col] = df_out.groupby('ESN')[col].ffill().fillna(0)
        else:
            df_out[col] = df_out[col].ffill().fillna(0)
    return df_out

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def apply_feature_engineering(df):
    """
    Orchestratore della pipeline di preparazione dati.
    Integra le logiche di normalizzazione nomenclatura, segmentazione,
    calcolo residui fisici e pulizia finale per l'input al modello ML.
    """
    df = df.copy()

    # Normalizzazione Schema Dati (Data Schema Standardization)
    map_cols = {'cycle': 'Snapshot', 'Cycle': 'Snapshot', 'time': 'Snapshot'}
    if 'Cycles' in df.columns and 'Cycles_Since_New' not in df.columns:
        map_cols['Cycles'] = 'Cycles_Since_New'
    
    # Mapping sensori standard
    sensor_map = {
        'P25': 'Sensed_P25', 'T3': 'Sensed_T3', 'T45': 'Sensed_T45', 
        'Ps3': 'Sensed_Ps3', 'Core_Speed': 'Sensed_Core_Speed'
    }
    for k, v in sensor_map.items():
        if k in df.columns and v not in df.columns:
            map_cols[k] = v
            
    df.rename(columns=map_cols, inplace=True)
    
    # Generazione contatore cicli sintetico se assente.
    if 'Cycles_Since_New' not in df.columns and 'ESN' in df.columns:
         df['Cycles_Since_New'] = df.groupby('ESN').cumcount() // 100

    # Esecuzione moduli pipeline
    if 'Sensed_Altitude' in df.columns:
        df = segment_flight_phases(df)
    else:
        df['Flight_Phase'] = 'CRUISE'

    df = detect_operational_profile(df)
    df = calculate_physics_residuals(df)
    df = apply_smoothing(df, window=10)
    
    # Filtraggio Fasi di Volo non significative per l'analisi di trend (es. Taxi, Landing).
    phases_ok = ['TAKEOFF', 'CLIMB', 'CRUISE']
    df = df[df['Flight_Phase'].isin(phases_ok)].copy()
    if df.empty: return df

    # One-Hot Encoding delle fasi di volo.
    df['Flight_Phase'] = pd.Categorical(df['Flight_Phase'], categories=phases_ok)
    dummies = pd.get_dummies(df['Flight_Phase'], prefix='Phase', dtype=int)
    df = pd.concat([df, dummies], axis=1)

    # Propagazione Forward (FFill) dei residui per garantire densità dati.
    feats_sensors = ['HPT_Resid_Smooth', 'HPC_Resid_Smooth', 'Global_Resid_Smooth', 
                     'Sensed_T45', 'Sensed_Core_Speed', 'Sensed_WFuel', 'Sensed_Ps3']
    for col in feats_sensors:
        if col in df.columns:
            if 'ESN' in df.columns:
                df[col] = df.groupby('ESN')[col].ffill()
            else:
                df[col] = df[col].ffill()

    # Encoding variabili categoriche residue.
    for col in df.select_dtypes(include=['category']).columns:
        try: df[col] = df[col].cat.codes
        except: df[col] = df[col].astype(str)
            
    df.fillna(0, inplace=True)
    
    # Ordinamento e selezione feature finale (Feature Selection).
    sort_cols = ['ESN', 'Cycles_Since_New']
    if 'Snapshot' in df.columns: sort_cols.append('Snapshot')
    df = df.sort_values(sort_cols)
    
    # Dropping colonne grezze o intermedie non richieste per l'inferenza.
    cols_to_drop = [
        'Flight_Progress', 'Flight_Phase', 
        'Sensed_P25', 'Sensed_T5', 'Sensed_VBV', 'Sensed_P2', 'Sensed_T2',
        'Sensed_Pamb', 'Sensed_Pt2', 'Sensed_VAFN', 'Sensed_Fan_Speed', 
        'Sensed_T25', 'Sensed_T3',
        'HPC_Resid', 'HPT_Resid', 'Global_Resid'
    ]
    cols_actual_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_actual_drop: df.drop(columns=cols_actual_drop, inplace=True)

    return df