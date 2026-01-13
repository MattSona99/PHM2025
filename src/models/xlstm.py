import torch
import torch.nn as nn

# ==========================================
# 0. GESTIONE IMPORT SICURO XLSTM
# ==========================================
# Gestione robusta delle dipendenze per l'architettura Extended LSTM.
# I moduli vengono inizializzati a None per consentire la definizione della classe
# anche in ambienti privi dei kernel CUDA ottimizzati necessari per xLSTM,
# delegando il controllo dell'errore alla fase di istanziazione.
xLSTMBlockStack = None
xLSTMBlockStackConfig = None
mLSTMBlockConfig = None
mLSTMLayerConfig = None
sLSTMBlockConfig = None
sLSTMLayerConfig = None

try:
    from xlstm import (
        xLSTMBlockStack, xLSTMBlockStackConfig,
        mLSTMBlockConfig, mLSTMLayerConfig,
        sLSTMBlockConfig, sLSTMLayerConfig
    )
except (OSError, ImportError, RuntimeError, Exception):
    pass

class Model_xLSTM(nn.Module):
    """
    Implementazione dell'architettura Extended LSTM (xLSTM) per la stima del RUL.
    Questa topologia supera i limiti delle LSTM convenzionali introducendo memorie matriciali (mLSTM)
    per aumentare la capacità di storage delle dipendenze a lungo termine e memorie scalari (sLSTM)
    per il gating e la stabilità del segnale.
    """
    def __init__(self, n_features, n_engines, n_profiles, n_targets=3,
                 emb_dim=4, hidden_size=128, num_layers=2, dropout=0.2, seq_len=50):
        super(Model_xLSTM, self).__init__()

        # Verifica di runtime della disponibilità delle librerie backend.
        if xLSTMBlockStack is None:
            raise RuntimeError("xLSTM library not found/loaded.")

        # --- FUSIONE DATI ETEROGENEI ---
        # Embedding vettoriale dei metadati categorici statici (Asset ID e Profilo Operativo).
        # Permette di proiettare informazioni discrete in uno spazio continuo denso,
        # arricchendo la serie temporale con il contesto operativo dell'asset.
        self.emb_esn = nn.Embedding(n_engines + 1, emb_dim) # +1 per gestione token OOV (Out-Of-Vocabulary)
        self.emb_prof = nn.Embedding(n_profiles + 1, emb_dim)

        self.input_dim_total = n_features + emb_dim * 2
        
        # Proiezione lineare per allineare la dimensionalità dello spazio delle feature (Input Space)
        # con la dimensionalità latente del blocco ricorrente (Hidden Space).
        self.input_projection = nn.Linear(self.input_dim_total, hidden_size)

        # --- CONFIGURAZIONE BLOCCO RICORRENTE ---
        # Definizione dello stack residuo xLSTM.
        # Si adotta una configurazione ibrida:
        # 1. mLSTM (Matrix LSTM): Utilizzata nei layer inferiori per massimizzare la capacità 
        #    di recupero delle informazioni storiche (Retrieval) grazie alla regola di update matriciale.
        # 2. sLSTM (Scalar LSTM): Posizionata nell'ultimo blocco per stabilizzare la dinamica 
        #    dello stato e fornire meccanismi di gating selettivo prima del regressore.
        cfg = xLSTMBlockStackConfig(
            embedding_dim=hidden_size,
            num_blocks=num_layers,
            context_length=seq_len,
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4)),
            slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="cuda")),
            slstm_at=[num_layers - 1] # Iniezione sLSTM al top dello stack
        )
        self.rnn = xLSTMBlockStack(cfg)

        # --- REGRESSORE DI OUTPUT ---
        # Multi-Layer Perceptron (MLP) per la mappatura dallo stato latente finale
        # allo spazio dei target continui (Remaining Useful Life per i vari componenti).
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_targets)
        )
        
        # Inizializzazione controllata dei pesi per favorire la convergenza (Kaiming He).
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_sens, x_esn, x_prof, return_features=False):
        batch_size, seq_len, _ = x_sens.size()

        # Espansione temporale degli embedding statici.
        # I vettori di contesto (ESN, Profilo) vengono replicati per ogni step temporale 't'
        # della finestra di osservazione, permettendo la concatenazione con i dati sensore variabili.
        vec_esn = self.emb_esn(x_esn).unsqueeze(1).expand(-1, seq_len, -1)
        vec_prof = self.emb_prof(x_prof).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenazione lungo l'asse delle feature (channel-wise).
        x = torch.cat((x_sens, vec_esn, vec_prof), dim=2)
        
        # Mapping nello spazio latente.
        x = self.input_projection(x)

        # Elaborazione sequenziale tramite stack xLSTM.
        x = self.rnn(x)
        
        # Strategia Many-to-One.
        # Si estrae lo stato nascosto corrispondente all'ultimo step temporale (t=T),
        # che condensa la storia evolutiva della degradazione all'interno della finestra.
        last_step = x[:, -1, :]

        if return_features:
            return last_step

        # Stima finale dei target.
        return self.head(last_step)