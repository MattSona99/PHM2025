import torch
import torch.nn as nn

class Model_DLinear(nn.Module):
    """
    Implementazione dell'architettura DLinear (Decomposition Linear) ottimizzata per il PHM.
    Si basa sul principio di decomposizione della serie temporale in due componenti ortogonali:
    1. Trend: Dinamica di degradazione a lungo termine (Monotonic Degradation Path).
    2. Seasonality/Residual: Fluttuazioni cicliche o rumore ad alta frequenza (Operating Condition variations).
    
    A differenza dei Transformer, DLinear utilizza mapping lineari diretti per proiettare
    le componenti decomposte, offrendo robustezza e ridotta complessità computazionale O(L).
    """

    def __init__(self, n_features, n_engines, n_profiles, n_targets=3,
                 window_size=120, hidden_size=None, dropout=0.3):
        super().__init__()
        
        # Normalizzazione istantanea per canale (Instance Normalization).
        # Essenziale per mitigare il covariate shift tra finestre temporali diverse
        # e standardizzare la scala dei sensori eterogenei (Pressioni, Temperature, RPM).
        self.input_norm = nn.InstanceNorm1d(n_features)
        
        # Branch di decomposizione parallela.
        # Si applicano trasformazioni lineari indipendenti per estrarre:
        # - Trend: Catturato tramite low-pass filtering implicito (weights averaging).
        # - Seasonality: Catturato tramite band-pass filtering implicito.
        # Input shape: [Window_Size] -> Output scalar: [1] (Compressione temporale)
        self.linear_trend = nn.Linear(window_size, 1)
        self.linear_season = nn.Linear(window_size, 1)
        
        # Dropout applicato post-decomposizione per regolarizzare l'apprendimento
        # delle due dinamiche ed evitare l'overfitting su pattern spuri.
        self.dropout_trend = nn.Dropout(dropout)
        self.dropout_season = nn.Dropout(dropout)
        
        # Layer Normalization applicata al vettore concatenato delle componenti estratte
        # per stabilizzare la distribuzione prima del fusion layer.
        self.norm_decomp = nn.LayerNorm(n_features * 2)
        
        # Embedding dei metadati statici.
        self.emb_esn = nn.Embedding(n_engines + 1, 4)
        self.emb_prof = nn.Embedding(n_profiles + 1, 4)
        self.dropout_emb = nn.Dropout(dropout / 2)
        
        # Testata di fusione profonda (Deep Fusion Head).
        # Aggrega le feature dinamiche decomposte con le feature statiche di contesto.
        # Input dim = (N_Features * 2 [Trend+Season]) + 8 [Embeddings]
        fusion_input_dim = n_features * 2 + 8
        
        self.fusion_proj = nn.Linear(fusion_input_dim, 128)
        
        self.fusion_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            
            # Output Layer con attivazione Softplus per garantire RUL >= 0
            # (Vincolo fisico di non-negatività della vita residua).
            nn.Linear(64, n_targets),
            nn.Softplus()
        )
        
        # Inizializzazione controllata dei pesi (Xavier Uniform).
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x_sens, x_esn, x_prof, return_features=False):
        # Permutazione tensore per compatibilità con InstanceNorm1d (Channel-First).
        # Input: [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        x = x_sens.transpose(1, 2)
        x = self.input_norm(x)
        
        # Decomposizione spettrale approssimata tramite proiezioni lineari.
        # Riduzione dimensionale: [batch, n_features, seq_len] -> [batch, n_features]
        trend = self.linear_trend(x).squeeze(-1)  # Estrazione componente a bassa frequenza
        season = self.linear_season(x).squeeze(-1)  # Estrazione componente ad alta frequenza
        
        # Regolarizzazione stocastica
        trend = self.dropout_trend(trend)
        season = self.dropout_season(season)
        
        # Concatenazione feature spettrali
        decomp = torch.cat([trend, season], dim=1)  # [batch, n_features*2]
        decomp = self.norm_decomp(decomp)
        
        # Recupero vettori di contesto statico
        v_esn = self.emb_esn(x_esn)    # [batch, 4]
        v_prof = self.emb_prof(x_prof)  # [batch, 4]
        v_emb = torch.cat([v_esn, v_prof], dim=1)  # [batch, 8]
        v_emb = self.dropout_emb(v_emb)
        
        # Costruzione Feature Vector globale
        feat = torch.cat([decomp, v_emb], dim=1)  # [batch, n_features*2 + 8]
        
        if return_features:
            return feat
        
        # Proiezione nello spazio latente di fusione
        x = self.fusion_proj(feat)
        
        # Stima finale dei target RUL
        output = self.fusion_head(x)
        
        return output