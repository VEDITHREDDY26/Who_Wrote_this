# Who Wrote This? (The Turing Detectives)

CAP 5610 Machine Learning, UCF, Fall 2026 (Dr. Mengxin Zheng)

**Team:** Vedith Reddy Kommula (MLP + Transformer), Shresta Reddy Saireddy (CNN + LSTM)

We classify who wrote a text passage: a human, or one of five LLM families (GPT, Llama, Mistral, MPT, Cohere). We use the clean, non-attacked split of the RAID benchmark (Dugan et al., ACL 2024). Four models are compared under identical conditions: an MLP on TF-IDF features, a TextCNN, a BiLSTM, and a fine-tuned DistilBERT. All four share the same seed, the same frozen group-aware splits, the same class weights and the same evaluation function. Macro-F1 is the main metric.

## Setup

### TODO: Mac (local smoke tests, CPU/MPS)

### TODO: Google Colab Pro (GPU)

### TODO: UCF Newton (SLURM, CUDA)

## Configuration

All paths live in `configs/config.yaml` and can be overridden with environment variables (`TD_DATA_DIR`, `TD_RAW_DIR`, `TD_SPLITS_DIR`, `TD_CHECKPOINT_DIR`, `TD_RESULTS_DIR`, `TD_CONFIG`, `TD_DEVICE`, `TD_SEED`). Check the resolved values with:

```bash
python -m src.config
```
