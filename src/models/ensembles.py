import torch
import torch.nn as nn
from .transformer import Model_Transformer
from .xlstm import Model_xLSTM

class Model_Ensemble_DLinear_Transformer(nn.Module):
    """
    Architettura ibrida a 'Mixture of Experts' con Gating Network dinamico.
    Si accoppia la robustezza dei modelli di decomposizione lineare (DLinear), eccellenti nel
    tracciare trend monotonici di degradazione, con la capacità rappresentativa dei Transformer,
    idonei a catturare dipendenze complesse non lineari.
    L'obiettivo è minimizzare l'errore di generalizzazione bilanciando stabilità e complessità.
    """
    def __init__(self, n_features, n_engines, n_profiles, n_targets=3,
                 hidden_size=128, nhead=4, num_layers=2, dropout=0.3, 
                 window_size=120, emb_dim=8):
        super().__init__()
        
        # Importazione locale per evitare dipendenze circolari durante l'inizializzazione del modulo.
        from .transformer import Model_Transformer
        from .dlinear import Model_DLinear
        
        # 1. Istanziazione Branch Transformer (High Variance / Low Bias potential)
        # Delegato alla cattura di pattern transitori e correlazioni globali.
        self.transformer = Model_Transformer(
            n_features, n_engines, n_profiles, n_targets, 
            emb_dim=emb_dim, hidden_size=hidden_size, nhead=nhead, 
            num_layers=num_layers, dropout=dropout
        )
        
        # 2. Istanziazione Branch DLinear (Low Variance / High Bias potential)
        # Delegato alla stima della componente di trend dominante (Baseline fisica).
        self.dlinear = Model_DLinear(
            n_features, n_engines, n_profiles, n_targets,
            window_size=window_size, dropout=dropout
        )
        
        # 3. Rete di Gating (Arbitrator).
        # Modulo leggero per la stima dinamica dei coefficienti di miscelazione.
        # La rete valuta il contesto operativo medio (statistiche globali della finestra)
        # per assegnare maggiore autorità al modello ritenuto più affidabile per lo specifico regime.
        self.gate_net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_targets), # Un peso di confidenza indipendente per ogni target (HPC, HPT, WW)
            nn.Sigmoid() # Output vincolato in [0, 1] per combinazione convessa
        )
        
    def forward(self, x_sens, x_esn, x_prof):
        # 1. Inferenza parallela sui rami esperti.
        # Si ottengono le stime indipendenti RUL dai due sottosistemi.
        pred_dlinear = self.dlinear(x_sens, x_esn, x_prof, return_features=False)
        pred_trans = self.transformer(x_sens, x_esn, x_prof, return_features=False)
        
        # 2. Calcolo del vettore di contesto per il Gating.
        # Si collassa la dimensione temporale calcolando la media per ottenere 
        # un'impronta statica del punto operativo corrente.
        x_mean = x_sens.mean(dim=1) # [batch, n_features]
        
        # 3. Determinazione dei pesi di fusione.
        gate = self.gate_net(x_mean) # [batch, n_targets]
        
        # 4. Fusione Pesata Adattiva (Adaptive Weighted Averaging).
        # Output = (Weight * Stable_Model) + ((1-Weight) * Complex_Model).
        # Il sistema apprende autonomamente a "fidarsi" del DLinear in regimi stabili
        # e del Transformer in regimi altamente dinamici o rumorosi.
        output = (gate * pred_dlinear) + ((1 - gate) * pred_trans)
        
        return output
    
class Model_Ensemble_xLSTM_Transformer(nn.Module):
    """
    Ensemble basato su 'Feature Fusion' (Late Fusion).
    A differenza dell'approccio precedente, qui non si combinano gli output finali,
    ma si concatenano gli spazi latenti generati da architetture ricorrenti (xLSTM) 
    e attenzionali (Transformer).
    Si sfrutta la complementarità tra l'elaborazione sequenziale locale (RNN) 
    e l'attenzione globale (Attention mechanism).
    """
    def __init__(self, n_features, n_engines, n_profiles, n_targets=3,
                 emb_dim=4, hidden_size=128, num_layers=2, dropout=0.2, 
                 pretrained_xlstm=None, pretrained_transformer=None):
        super(Model_Ensemble_xLSTM_Transformer, self).__init__()
        
        # Gestione modulare dei backbone.
        # Si prevede la possibilità di iniettare pesi pre-addestrati (Transfer Learning)
        # o inizializzare nuove istanze per l'addestramento end-to-end.
        if pretrained_xlstm: self.model_xlstm = pretrained_xlstm
        else: self.model_xlstm = Model_xLSTM(n_features, n_engines, n_profiles, n_targets, emb_dim, hidden_size, num_layers, dropout)

        if pretrained_transformer: self.model_transformer = pretrained_transformer
        else: self.model_transformer = Model_Transformer(n_features, n_engines, n_profiles, n_targets, emb_dim, hidden_size, 4, num_layers, dropout)

        # Definizione della Testata di Fusione (Fusion Head).
        # MLP profondo che mappa lo spazio delle feature congiunto (dimensione 2*Hidden)
        # sui target di prognostica finali.
        fusion_dim = hidden_size * 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128), # Normalizzazione per stabilizzare l'input al ReLU
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_targets)
        )

    def forward(self, x_sens, x_esn, x_prof):
        # 1. Estrazione delle rappresentazioni latenti (Feature Extraction).
        # Si recuperano i vettori di contesto dall'ultimo layer degli encoder,
        # ignorando le teste di predizione originali dei singoli modelli.
        feat_xlstm = self.model_xlstm(x_sens, x_esn, x_prof, return_features=True)
        feat_transf = self.model_transformer(x_sens, x_esn, x_prof, return_features=True)
        
        # 2. Concatenazione dei vettori di feature.
        # Si crea una rappresentazione arricchita che include sia la memoria a breve termine (xLSTM)
        # che le relazioni a lungo termine (Transformer).
        combined = torch.cat([feat_xlstm, feat_transf], dim=1)
        
        # 3. Stima congiunta RUL.
        return self.fusion_head(combined)