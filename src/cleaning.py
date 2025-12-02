import pandas as pd
import numpy as np

SENSORS_CONFIG = [
    # --- CONDIZIONI OPERATIVE (Pulizia Simmetrica Standard) ---
    {'name': 'Sensed_Altitude', 'min_phys': 0, 'max_phys': 50000, 'iqr_direction': 'both', 'iqr_factor': 2.5},
    {'name': 'Sensed_Mach',     'min_phys': 0.0, 'max_phys': 1.0, 'iqr_direction': 'both', 'iqr_factor': 2.0},
    {'name': 'Sensed_Pamb',     'min_phys': 0.1, 'max_phys': 16.0, 'iqr_direction': 'both', 'iqr_factor': 2.0},
    {'name': 'Sensed_Pt2',      'min_phys': 0.1, 'max_phys': 25.0, 'iqr_direction': 'both', 'iqr_factor': 2.0},
    {'name': 'Sensed_TAT',      'min_phys': 350.0, 'max_phys': 650.0, 'iqr_direction': 'both', 'iqr_factor': 2.0},

    # --- SENSORI CON SPIKE VERSO L'ALTO (Upper Only) ---
    {'name': 'Sensed_WFuel',    'min_phys': 0.0, 'max_phys': 2.0, 'iqr_direction': 'upper', 'iqr_factor': 2.0},
    {'name': 'Sensed_T25',      'min_phys': 0.0, 'max_phys': 2000.0, 'iqr_direction': 'upper', 'iqr_factor': 2.0},
    {'name': 'Sensed_T3',       'min_phys': 1000.0, 'max_phys': 2000.0, 'iqr_direction': 'upper', 'iqr_factor': 1.5}, # Nota: T3 aveva anche drop <1000
    {'name': 'Sensed_T45',      'min_phys': 1500.0, 'max_phys': 2400.0, 'iqr_direction': 'upper', 'iqr_factor': 2.0},
    {'name': 'Sensed_T5',       'min_phys': 0.0, 'max_phys': 2000.0, 'iqr_direction': 'upper', 'iqr_factor': 2.0},
    {'name': 'Sensed_Core_Speed', 'min_phys': 0.0, 'max_phys': 22000.0, 'iqr_direction': 'both', 'iqr_factor': 1.5}, # N2 aveva anche simmetria

    # --- SENSORI CON DROP VERSO IL BASSO (Lower Only) ---
    {'name': 'Sensed_P25',      'min_phys': 15.0, 'max_phys': 1000.0, 'iqr_direction': 'lower', 'iqr_factor': 1.5},
    {'name': 'Sensed_Ps3',      'min_phys': 150.0, 'max_phys': 1000.0, 'iqr_direction': 'lower', 'iqr_factor': 1.5},
    {'name': 'Sensed_Fan_Speed','min_phys': 100.0, 'max_phys': 5000.0, 'iqr_direction': 'lower', 'iqr_factor': 1.5},
    {'name': 'Sensed_VAFN',     'min_phys': 0.0, 'max_phys': 10000.0, 'iqr_direction': 'lower', 'iqr_factor': 2.0},

    # --- SENSORI SPECIALI (No IQR) ---
    {'name': 'Sensed_VBV',      'min_phys': 0.0, 'max_phys': 5.0, 'skip_iqr': True}
]

def clean_sensor(df_input, sensor_config):
    """
    Funzione per la pulizia dei dati dei sensori.
    Applica Hard Bounds, IQR Cleaning direzionale e Interpolazione
    basandosi su un dizionario di configurazione.
    """
    df = df_input.copy()
    sensor = sensor_config['name']

    if sensor not in df.columns:
        return df

    print(f"--- Pulizia {sensor} ---")

    # --- 1. HARD BOUNDS (Limiti Fisici) ---
    min_phys = sensor_config.get('min_phys', -np.inf)
    max_phys = sensor_config.get('max_phys', np.inf)

    mask_phys = (df[sensor] < min_phys) | (df[sensor] > max_phys)
    count_phys = mask_phys.sum()
    
    if count_phys > 0:
        print(f"   -> Rimossi {count_phys} valori impossibili (fuori {min_phys} - {max_phys}).")
        df.loc[mask_phys, sensor] = np.nan

    # --- 2. STATISTICA ROBUSTA (IQR per Motore) ---
    if not sensor_config.get('skip_iqr', False):
        
        direction = sensor_config.get('iqr_direction', 'both')
        factor = sensor_config.get('iqr_factor', 1.5)

        for esn in df['ESN'].unique():
            esn_mask = df['ESN'] == esn
            series = df.loc[esn_mask, sensor]

            # Calcolo Quartili sui dati validi (già puliti dai limiti fisici)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            # Determina Outlier Mask in base alla direzione
            outlier_mask = pd.Series(False, index=series.index)

            if direction in ['lower', 'both']:
                lower_bound = Q1 - factor * IQR
                outlier_mask |= (series < lower_bound)
            
            if direction in ['upper', 'both']:
                upper_bound = Q3 + factor * IQR
                outlier_mask |= (series > upper_bound)

            count_stat = outlier_mask.sum()
            if count_stat > 0:
                df.loc[esn_mask & outlier_mask, sensor] = np.nan

    # --- 3. INTERPOLAZIONE ---
    # Riempie i buchi creati (sia fisici che statistici)
    df[sensor] = df.groupby('ESN')[sensor].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

    return df

def clean_all_sensors(df):
    df_cleaned = df.copy()
    for config in SENSORS_CONFIG:
        df_cleaned = clean_sensor(df_cleaned, config)
    return df_cleaned