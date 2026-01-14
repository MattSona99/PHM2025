# Predictive Maintenance (PdM) Framework per Stima RUL

## 1. Panoramica del Progetto
Questo repository ospita una pipeline completa di **Deep Learning** progettata per la **Manutenzione Predittiva** di motori di aerei per la challenge "PHM North America 2025 Conference Data Challenge". Il sistema analizza serie temporali dei cicli di volo per stimare la Vita Residua Utile (**RUL - Remaining Useful Life**) di componenti critici, supportando strategie di manutenzione proattiva.

Il framework è in grado di predire simultaneamente tre target di degradazione:
1.  **HPC_SV:** Stato di salute del Compressore ad Alta Pressione.
2.  **HPT_SV:** Stato di salute della Turbina ad Alta Pressione.
3.  **WW:** Lavaggio del Compressore (Water-Wash).

## 2. Metodologia e Architettura

Il flusso di lavoro implementa una pipeline per il trattamento di serie temporali, strutturato nelle seguenti fasi:

### 2.1 Preprocessing & Feature Engineering
* **Physics-Based Residuals:** Calcolo dei residui termodinamici basati su equazioni di stato per isolare le anomalie dai cambiamenti di regime operativo.
* **Flight Phase Segmentation:** Segmentazione automatica delle fasi di volo (Takeoff, Climb, Cruise) per filtrare dati non stazionari.
* **Signal Smoothing:** Applicazione di medie mobili per la riduzione del rumore di misura ad alta frequenza.
* **Pipeline Scikit-Learn:** Scaling robusto (`RobustScaler`) per i sensori e normalizzazione (`MinMaxScaler`) per i target, integrati in una pipeline serializzabile.

### 2.2 Architetture Neurali
Il sistema supporta modelli modulari e strategie di Ensemble:

* **Transformer Encoder:** Utilizza *Multi-Head Self-Attention* e *Learnable Positional Encodings* per catturare dipendenze temporali a lungo raggio. Include un layer di *Attention Pooling* per l'aggregazione temporale pesata.
* **xLSTM (Extended LSTM):** Implementa blocchi *mLSTM* (Matrix Memory) e *sLSTM* (Scalar Memory) per superare i limiti di capacità delle RNN tradizionali.
* **DLinear:** Architettura basata sulla decomposizione del segnale in componenti di *Trend* (bassa frequenza) e *Stagionalità/Residuo* (alta frequenza), processate da layer lineari indipendenti.
* **Ensemble Ibridi:**
    * **MoE (Mixture of Experts):** Fusione DLinear + Transformer regolata da un Gating Network dinamico che pesa i contributi in base al contesto operativo.
    * **Feature Fusion:** Concatenazione degli spazi latenti di xLSTM e Transformer.

### 2.3 Loss Function
Viene utilizzata una **Asymmetric Huber Loss**. Questa funzione di costo penalizza in modo non lineare le stime sovrastimate (RUL predetto > RUL reale), minimizzando il rischio di guasti inattesi ("Late Prediction Penalty").

## 3. Struttura della Repository

```text
.
├── data/
│   ├── raw/                 # File CSV di input (train/test)
│   ├── processed/           # Dati processati
│   ├── temp/                # Artefatti intermedi
│   └── results/             # Submission files e log di training
├── models/                  # Salvataggio pesi (.pth) e pipeline (.pkl)
├── src/
|   ├── models/
|   │   ├── models.py              # Init package per importazione modelli
|   │   ├── layers.py              # Layer custom (Attention, Positional Encoding)
|   │   ├── dlinear.py             # Implementazione DLinear
|   │   ├── transformer.py         # Implementazione Transformer
|   │   └── xlstm.py               # Implementazione xLSTM
│   ├── configs.py                 # Dizionari di configurazione iperparametri
│   ├── feature_engineering.py     # Logica di dominio (fisica del motore)
│   ├── preprocessing.py           # Pipeline di trasformazione dati
├── train.py                 # Script principale di addestramento
├── test.py                  # Script di inferenza e generazione output
├── requirements.txt         # Dipendenze Python
└── README.md                # Documentazione di progetto
```

## 4. Installazione e Requisiti

### Prerequisiti
* Python 3.8+
* NVIDIA CUDA Toolkit (verificato su 11.8 / 12.1)
* Compilatore MSVC (per ambienti Windows)

### Setup
1.  Clonare il repository.
2.  Installare le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Nota Windows:** Lo script `train.py` include una patch per montare il path CUDA su un drive virtuale `Z:` per aggirare i limiti di lunghezza dei path di Windows durante la compilazione JIT di PyTorch. Assicurarsi che l'ambiente supporti il comando `subst`.

## 5. Utilizzo

### 5.1 Training (`train.py`)
Lo script gestisce l'intero ciclo di vita dell'addestramento:
1.  **Cross-Validation:** Esegue una validazione Leave-One-Group-Out basata su ESN (Engine Serial Number) per valutare la robustezza.
2.  **Full Retraining:** Addestra il modello finale sull'intero dataset per la produzione.

**Configurazione:** Modificare la lista `MODELS_TO_RUN` in `train.py` o i parametri in `src/configs.py`.

```bash
python train.py
```

### 5.2 Inferenza (`test.py`)
Carica il modello finale e la pipeline di preprocessing per generare predizioni su nuovi dati.

1.  Assicurarsi che `train_full_model` sia stato eseguito (richiede `final_model.pth` e `final_pipeline.pkl`).
2.  Impostare la variabile `MODEL_TO_TEST` in `test.py`.

```bash
python test.py
```
Il file di output sarà generato in `data/results/{model_name}/submission_final.csv`.

## 6. Metriche di Valutazione

Le performance sono valutate tramite uno **Score Competizione Pesato**:

$$Score = \text{mean}(w(t) \cdot (y_{pred} - y_{true})^2)$$

Dove il peso $w(t)$ cresce esponenzialmente man mano che il componente si avvicina al fine vita, privilegiando la precisione nelle fasi critiche.

---
**Licenza:** MIT
