import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder

class Pipeline:
    """
    Orchestratore per la normalizzazione, codifica e segmentazione temporale
    dei dati di telemetria. Gestisce la trasformazione da dati tabulari grezzi
    a tensori sequenziali idonei per reti neurali ricorrenti o Transformer.
    """
    def __init__(self,
                 windows_size=50,
                 feature_cols=None,
                 target_cols=['Cycles_to_HPC_SV', 'Cycles_to_HPT_SV', 'Cycles_to_WW'],
                 meta_cols=['ESN', 'Operational_Profile'],
                 cycle_col='Cycles_Since_New',
                 snapshot_col='Snapshot'):
        
        self.window_size = windows_size
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.meta_cols = meta_cols
        self.cycle_col = cycle_col
        self.snapshot_col = snapshot_col
        
        # Adozione di RobustScaler per le variabili di sensore al fine di mitigare
        # l'impatto di outlier e rumore di misura non gaussiano tipico della strumentazione di campo.
        self.scaler = RobustScaler()
        
        # Utilizzo di MinMaxScaler per i target (RUL) per confinare il dominio di output
        # nell'intervallo [0, 1], favorendo la stabilità numerica durante la backpropagation.
        self.target_scaler = MinMaxScaler() 
        
        self.esn_encoder = LabelEncoder()
        self.profile_encoder = LabelEncoder()
        
        self.n_features = 0
        self.n_engines = 0
        self.n_profiles = 0
        self.n_targets = len(self.target_cols)
        
    def fit(self, df):
        print("--- Fitting Preprocessing Pipeline ---")
        df = df.copy()
        
        # Procedura di identificazione automatica delle feature informative.
        # Si escludono metadati, identificativi e target per isolare le sole variabili di stato.
        if self.feature_cols is None:
            # 1. Definizione lista di esclusione per colonne strutturali e di fase di volo.
            exclude_exact = self.meta_cols + self.target_cols + \
                            [self.snapshot_col, 'Flight_Phase', self.cycle_col, 'Cycles', 
                             'Flight_Progress', 'Phase_TAKEOFF', 'Phase_CLIMB', 'Phase_CRUISE']
            
            # 2. Esclusione basata su pattern matching per evitare Data Leakage
            # rimuovendo feature calcolate o label non dichiarate.
            exclude_keywords = ['Cumulative', 'RUL', 'label', 'target', 'Unnamed']
            
            self.feature_cols = []
            for c in df.columns:
                # Verifica appartenenza alla lista di esclusione esatta
                if c in exclude_exact: continue
                
                # Verifica presenza keyword proibite
                if any(k in c for k in exclude_keywords): continue
                
                # Verifica consistenza del tipo di dato (solo numerico ammesso)
                if not pd.api.types.is_numeric_dtype(df[c]): continue
                
                self.feature_cols.append(c)
            
        self.n_features = len(self.feature_cols)

        # Calcolo delle statistiche (mediana, IQR) per la normalizzazione dei sensori.
        self.scaler.fit(df[self.feature_cols])
        
        # Calcolo dei range (min, max) per la normalizzazione dei target.
        if all(c in df.columns for c in self.target_cols):
            self.target_scaler.fit(df[self.target_cols])
        
        # Mappatura dello spazio degli identificativi motore (ESN).
        self.esn_encoder.fit(df['ESN'])
        self.n_engines = len(self.esn_encoder.classes_)
        
        # Mappatura dei profili operativi se presenti, altrimenti gestione caso default.
        if 'Operational_Profile' in df.columns:
            df['Operational_Profile'] = df['Operational_Profile'].astype(str)
            self.profile_encoder.fit(df['Operational_Profile'])
            self.n_profiles = len(self.profile_encoder.classes_)
        else:
            self.profile_encoder.fit(['default'])
            self.n_profiles = 1
            
        return self
    
    def transform(self, df, stride=1):
        data = df.copy()
        
        # Verifica integrità dello schema dati in ingresso.
        # Eventuali feature mancanti (es. sensori guasti/assenti nel test set) vengono
        # riempite con 0.0 (valore neutro post-scaling robusto, approx. la mediana).
        missing_cols = [c for c in self.feature_cols if c not in data.columns]
        if missing_cols:
            for c in missing_cols:
                data[c] = 0.0

        # Applicazione della normalizzazione sullo spazio delle feature.
        data[self.feature_cols] = self.scaler.transform(data[self.feature_cols])
        
        has_targets = all(c in data.columns for c in self.target_cols)
        if has_targets:
            data[self.target_cols] = self.target_scaler.transform(data[self.target_cols])
        
        # Codifica ESN con gestione robusta dei livelli non osservati (Unknown Handling).
        # Gli ESN non presenti nel training set vengono mappati sul primo indice disponibile.
        known_esns = set(self.esn_encoder.classes_)
        default_esn = self.esn_encoder.classes_[0]
        data['ESN'] = data['ESN'].apply(lambda x: x if x in known_esns else default_esn)
        data['ESN_Enc'] = self.esn_encoder.transform(data['ESN'])
        
        # Codifica Profilo Operativo con logica analoga di fallback.
        if 'Operational_Profile' in data.columns:
            data['Operational_Profile'] = data['Operational_Profile'].astype(str)
            known_profs = set(self.profile_encoder.classes_)
            default_prof = self.profile_encoder.classes_[0]
            data['Operational_Profile'] = data['Operational_Profile'].apply(lambda x: x if x in known_profs else default_prof)
            data['Profile_Enc'] = self.profile_encoder.transform(data['Operational_Profile'])
        else:
            data['Profile_Enc'] = 0
            
        # Generazione delle sequenze temporali (Sliding Window).
        # Si trasformano le serie temporali continue in segmenti discreti di lunghezza fissa
        # per l'addestramento supervisionato.
        X_sensors, X_esn, X_prof, y_list = [], [], [], []
        unique_engines = data['ESN'].unique()
        
        

        for engine in unique_engines:
            # Ordinamento temporale rigoroso basato su Snapshot o Cicli progressivi.
            # Fondamentale per mantenere la causalità temporale dei dati.
            df_eng = data[data['ESN'] == engine].copy()
            
            if self.cycle_col in df_eng.columns and self.snapshot_col in df_eng.columns:
                cycles = df_eng[self.cycle_col].unique()
                snapshots = range(1,9)
                
                idx_ideal = pd.MultiIndex.from_product(
                    [cycles, snapshots],
                    names=[self.cycle_col, self.snapshot_col]
                )
                
                cols_static = ['ESN_Enc', 'Profile_Enc']
                cols_dynamic = [c for c in df_eng.columns if c not in cols_static + [self.cycle_col, self.snapshot_col]]
                
                df_eng = df_eng.set_index([self.cycle_col, self.snapshot_col])
                df_dynamic = df_eng[cols_dynamic].reindex(idx_ideal)
                df_dynamic = df_dynamic.groupby(level=0).ffill().bfill()
                df_dynamic = df_dynamic.ffill().fillna(0)
                
                df_clean = df_dynamic.reset_index()

                df_static_source = df_eng[cols_static].reset_index().drop_duplicates(subset=[self.cycle_col])

                df_static_source = df_static_source[[self.cycle_col] + cols_static]

                df_clean = df_clean.merge(df_static_source, on=self.cycle_col, how='left').ffill()

                df_clean = df_clean.sort_values([self.cycle_col, self.snapshot_col])
            
            else:
                sort_cols = [c for c in [self.cycle_col, 'time'] if c in df_eng.columns]
                if sort_cols:
                    df_clean = df_eng.sort_values(sort_cols)
                else:
                    df_clean = df_eng

            vals_sens = df_eng[self.feature_cols].values
            vals_esn = df_eng['ESN_Enc'].values
            vals_prof = df_eng['Profile_Enc'].values
            
            vals_targets = df_eng[self.target_cols].values if has_targets else None
            
            num_records = len(df_eng)
            
            # Iterazione con passo 'stride' per la creazione dei batch sequenziali.
            for i in range(0, num_records - self.window_size, stride):
                # Estrazione finestra sensori [t : t + window]
                X_sensors.append(vals_sens[i : i + self.window_size])
                
                # Associazione metadati statici riferiti all'ultimo step della finestra
                X_esn.append(vals_esn[i + self.window_size - 1])
                X_prof.append(vals_prof[i + self.window_size - 1])
                
                if has_targets:
                    y_list.append(vals_targets[i + self.window_size - 1]) 
        
        # Assemblaggio del dizionario di output con tensori NumPy ottimizzati (float32/int32).
        X_dict = {
            'input_sensors': np.array(X_sensors, dtype=np.float32),
            'input_esn': np.array(X_esn, dtype=np.int32),
            'input_profile': np.array(X_prof, dtype=np.int32)
        }
        
        y = np.array(y_list, dtype=np.float32) if y_list else None
        
        return X_dict, y
    
    def inverse_transform_y(self, y_scaled):
        """
        Inversione della trasformazione di scala sui target predetti.
        Restituisce i valori di RUL nel dominio originale (Cicli).
        """
        return self.target_scaler.inverse_transform(y_scaled)