import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Modulo per l'iniezione dell'informazione posizionale.
    Poiché l'architettura Transformer è per natura invariante rispetto alle permutazioni,
    si rende necessaria l'addizione di segnali sinusoidali a frequenze variabili per
    codificare l'ordine sequenziale dei dati di telemetria.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-calcolo della matrice posizionale statica.
        # Si utilizzano funzioni seno e coseno con frequenze geometricamente decrescenti
        # per permettere al modello di attendere facilmente a posizioni relative.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registrazione come buffer persistente ma non addestrabile (non aggiornato da gradient descent).
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Addizione element-wise dell'encoding all'embedding di input.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Variante adattiva dell'encoding posizionale.
    Invece di funzioni fisse, si allocano parametri addestrabili che permettono
    alla rete di apprendere la rappresentazione temporale ottimale per la specifica
    dinamica di degradazione del dataset.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Inizializzazione dei pesi con deviazione standard ridotta per non destabilizzare
        # la convergenza nelle prime epoche.
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """
    Meccanismo di aggregazione temporale basato su attenzione (Temporal Attention Pooling).
    Invece di utilizzare un semplice Global Average Pooling o prendere l'ultimo step,
    si calcola una media pesata degli step temporali, assegnando maggiore rilevanza
    agli istanti in cui il segnale presenta caratteristiche critiche per la stima del RUL.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentionPooling, self).__init__()
        
        # Rete feed-forward per il calcolo degli score di attenzione (Energy function).
        # Si utilizza l'attivazione GELU per migliori proprietà del gradiente rispetto a ReLU.
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        # Layer Normalization applicata pre-attivazione per stabilizzare la distribuzione delle feature.
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):

        # Normalizzazione input
        x_norm = self.layer_norm(x)
        
        # Calcolo logit di attenzione
        scores = self.attention(x_norm)  # [batch_size, seq_len, 1]
        
        # Mascheramento opzionale per gestire sequenze di lunghezza variabile (padding).
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        # Calcolo pesi normalizzati (distribuzione di probabilità sull'asse temporale).
        weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Aggregazione pesata per ottenere il vettore di contesto globale.
        context_vector = torch.sum(x * weights, dim=1)  # [batch_size, hidden_size]
        
        return context_vector


class MultiHeadAttentionPooling(nn.Module):
    """
    Aggregatore avanzato basato su Multi-Head Attention.
    Permette al modello di focalizzarsi su diversi sottospazi di rappresentazione
    contemporaneamente. Un vettore 'query' addestrabile interroga la sequenza
    per estrarre un sommario coerente dello stato di salute dell'asset.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadAttentionPooling, self).__init__()
        
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) deve essere divisibile per num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        
        # Definizione del vettore Query come parametro addestrabile (Pooling Query).
        # Agisce come un "probe" statico che impara a selezionare le feature rilevanti dalla sequenza.
        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim) * 0.02)
        
        # Proiezioni lineari per Key e Value (dallo spazio feature agli spazi delle head).
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Proiezione finale per ricomporre l'output.
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        x_norm = self.layer_norm(x)
        
        # Calcolo proiezioni K, V
        keys = self.key_proj(x_norm)  # [B, L, H]
        values = self.value_proj(x_norm)  # [B, L, H]
        
        # Riorganizzazione tensori per calcolo parallelo sulle head:
        # [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Espansione della Query per il batch corrente:
        # [1, num_heads, head_dim] -> [B, num_heads, 1, head_dim]
        query = self.query.expand(batch_size, -1, -1).unsqueeze(2)
        
        # Scaled Dot-Product Attention: Q * K^T / sqrt(d_k)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, num_heads, 1, L]
        
        # Applicazione maschera di padding
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Calcolo pesi di attenzione
        attn_weights = torch.softmax(scores, dim=-1)  # [B, num_heads, 1, L]
        attn_weights = self.dropout(attn_weights)
        
        # Aggregazione valori pesata
        context = torch.matmul(attn_weights, values)
        
        # Concatenazione delle head: [B, num_heads, 1, head_dim] -> [B, hidden_size]
        context = context.squeeze(2).contiguous().view(batch_size, self.hidden_size)
        
        # Proiezione lineare finale
        context = self.out_proj(context)
        
        return context