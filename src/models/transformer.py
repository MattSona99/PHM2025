import torch
import torch.nn as nn
from .layers import (
    PositionalEncoding, 
    LearnablePositionalEncoding,
    AttentionPooling,
    MultiHeadAttentionPooling
)


class Model_Transformer(nn.Module):
    """
    Implementazione di un'architettura basata su Transformer Encoder per l'analisi 
    di serie temporali multivariate. Il modello sfrutta il meccanismo di Self-Attention 
    per catturare dipendenze a lungo termine e relazioni non lineari nei dati di telemetria, 
    prescindendo dalla distanza temporale tra gli eventi.
    """
    
    def __init__(
        self, 
        n_features,
        n_engines,
        n_profiles,
        n_targets=3,
        emb_dim=4,
        hidden_size=128,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        learnable_pe=True,
        use_multihead_pooling=True,
        pooling_heads=4,
        **kwargs  # Parametri addizionali non pertinenti alla struttura del grafo computazionale
    ):
        super(Model_Transformer, self).__init__()
        
        # Validazione dei vincoli architetturali per Multi-Head Attention.
        assert hidden_size % nhead == 0, \
            f"Hidden size ({hidden_size}) deve essere divisibile per nhead ({nhead})"
        
        # Controllo adattivo sulla dimensionalità degli embedding.
        # Si garantisce una capacità rappresentativa minima per evitare colli di bottiglia informativi.
        if emb_dim < max(8, hidden_size // 32):
            emb_dim = max(8, hidden_size // 32)
            print(f"   [Transformer] Emb_dim aumentato automaticamente a {emb_dim}")
        
        self.d_model = hidden_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        
        # ========== EMBEDDINGS ==========
        # Proiezione di variabili categoriche (ESN, Profilo) in uno spazio vettoriale denso.
        # +1 negli indici gestisce il token OOV (Out-Of-Vocabulary) o padding.
        self.emb_esn = nn.Embedding(n_engines + 1, emb_dim)
        self.emb_prof = nn.Embedding(n_profiles + 1, emb_dim)
        
        # ========== INPUT PROJECTION ==========
        # Layer di adattamento per mappare la concatenazione di feature continue e embedding
        # nello spazio latente del Transformer (d_model). Include normalizzazione per stabilizzare il gradiente.
        self.input_projection = nn.Sequential(
            nn.Linear(n_features + emb_dim * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        # ========== POSITIONAL ENCODING ==========
        # Iniezione dell'informazione sull'ordine sequenziale, necessaria data la natura
        # permutation-invariant del meccanismo di self-attention.
        if learnable_pe:
            self.pos_encoder = LearnablePositionalEncoding(
                hidden_size, 
                max_len=5000, 
                dropout=dropout
            )
        else:
            self.pos_encoder = PositionalEncoding(
                hidden_size, 
                max_len=5000, 
                dropout=dropout
            )
        
        # ========== TRANSFORMER ENCODER ==========
        # Stack di blocchi Encoder standard. Ogni blocco comprende:
        # 1. Multi-Head Self-Attention
        # 2. Feed-Forward Network (Pointwise)
        # 3. Residual Connections e Layer Normalization (Pre-Norm architecture preferita per stabilità)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu', # Attivazione GELU per migliori proprietà di flusso del gradiente
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # ========== POOLING LAYER ==========
        # Aggregazione temporale per ridurre la sequenza a un singolo vettore di contesto.
        # Si utilizza un meccanismo di attenzione (Attention Pooling) per pesare dinamicamente
        # la rilevanza di ogni step temporale rispetto al task di predizione.
        if use_multihead_pooling:
            self.pooling = MultiHeadAttentionPooling(
                hidden_size, 
                num_heads=pooling_heads,
                dropout=dropout
            )
        else:
            self.pooling = AttentionPooling(hidden_size, dropout=dropout)
        
        # ========== PREDICTION HEAD ==========
        # Regressore finale (MLP) per mappare il vettore di contesto ai target continui (RUL).
        # L'uso di Softplus nell'ultimo layer impone un vincolo di positività sull'output,
        # coerente con la natura fisica della vita residua.
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(64, n_targets),
            nn.Softplus()
        )
        
        # Inizializzazione controllata dei parametri
        self._init_weights()

    def _init_weights(self):
        """
        Procedura di inizializzazione dei pesi specifica per layer.
        Si utilizza Xavier/Glorot initialization per bilanciare la varianza dei gradienti
        attraverso i layer profondi.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'head' in name:
                    # Gain ridotto per i layer finali per evitare output iniziali esplosivi
                    nn.init.xavier_normal_(module.weight, gain=0.5)
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.Embedding):
                # Embedding inizializzati con distribuzione normale a bassa varianza
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x_sens, x_esn, x_prof, padding_mask=None, return_features=False):
        """
        Definizione del flusso computazionale (Forward Pass).
        """
        batch_size, seq_len, _ = x_sens.size()
        
        # ========== EMBEDDING ==========
        # Espansione temporale degli embedding statici per allineamento con la serie temporale
        vec_esn = self.emb_esn(x_esn).unsqueeze(1).expand(-1, seq_len, -1)
        vec_prof = self.emb_prof(x_prof).unsqueeze(1).expand(-1, seq_len, -1)
        
        # ========== CONCATENAZIONE E PROIEZIONE ==========
        # Fusione feature dinamiche e statiche lungo la dimensione dei canali
        x = torch.cat((x_sens, vec_esn, vec_prof), dim=2)
        
        # Proiezione nello spazio latente
        x = self.input_projection(x)
        
        # ========== POSITIONAL ENCODING ==========
        # Somma dell'informazione posizionale
        x = self.pos_encoder(x)
        
        # ========== TRANSFORMER ENCODING ==========
        # Elaborazione tramite self-attention layers.
        # src_key_padding_mask viene passato per ignorare i token di padding nel calcolo dell'attenzione.
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # ========== POOLING ==========
        # Compressione della dimensione temporale
        x_pooled = self.pooling(x, mask=padding_mask)
        
        if return_features:
            return x_pooled
        
        # ========== PREDICTION HEAD ==========
        output = self.head(x_pooled)
        
        return output

    def get_attention_weights(self, x_sens, x_esn, x_prof, padding_mask=None):
        """
        Metodo di utilità per l'estrazione dei pesi di attenzione (Interpretabilità).
        Permette di visualizzare su quali parti della sequenza temporale il modello
        si sta focalizzando per generare la predizione.
        """
        batch_size, seq_len, _ = x_sens.size()
        
        vec_esn = self.emb_esn(x_esn).unsqueeze(1).expand(-1, seq_len, -1)
        vec_prof = self.emb_prof(x_prof).unsqueeze(1).expand(-1, seq_len, -1)
        
        x = torch.cat((x_sens, vec_esn, vec_prof), dim=2)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Calcolo manuale degli score di attenzione dall'ultimo layer di pooling
        x_norm = self.pooling.layer_norm(x)
        scores = self.pooling.attention(x_norm).squeeze(-1)
        
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=1)
        
        return attention_weights