# Transformer Implementation

A PyTorch implementation of the Transformer architecture for neural machine translation from scratch, based on the "Attention Is All You Need" paper.

## Overview

This project implements a complete Transformer model for translating between English and Swedish using the OPUS Books dataset. The implementation includes all core components: multi-head attention, positional encoding, encoder-decoder architecture, and training pipeline.

## Features

- Complete Transformer architecture implementation from scratch
- Multi-head self-attention and cross-attention mechanisms
- Sinusoidal positional encoding
- Layer normalization and residual connections
- Custom tokenizer training for source and target languages
- Comprehensive training loop with validation metrics
- TensorBoard logging for training visualization
- Greedy decoding for inference

## Project Structure

```
transformer/
├── model.py          # Transformer architecture implementation
├── train.py          # Training script and validation logic
├── dataset.py        # Dataset class and data preprocessing
├── config.py         # Configuration parameters
├── requirements.txt  # Python dependencies
├── tokenizer_en.json # English tokenizer (generated)
├── tokenizer_sv.json # Swedish tokenizer (generated)
├── weights/          # Model checkpoints directory
└── runs/             # TensorBoard logs directory
```

## Requirements

- Python 3.9
- PyTorch 2.0.1
- See `requirements.txt` for complete dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training script:
```bash
python train.py
```

The script will:
- Download and prepare the OPUS Books English-Swedish dataset
- Build or load tokenizers for both languages
- Train the Transformer model
- Save checkpoints in the `weights/` directory
- Log training metrics to TensorBoard

### Configuration

Modify `config.py` to adjust training parameters:

- `batch_size`: Training batch size (default: 2)
- `num_epochs`: Number of training epochs (default: 5)
- `lr`: Learning rate (default: 1e-4)
- `seq_len`: Maximum sequence length (default: 320)
- `d_model`: Model dimension (default: 128)
- `lang_src`: Source language (default: "en")
- `lang_tgt`: Target language (default: "sv")

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir=runs
```

## Model Architecture

The implementation includes:

### Core Components
- **InputEmbeddings**: Token embedding with scaling
- **PositionalEncoding**: Sinusoidal position embeddings
- **MultiHeadAttentionBlock**: Multi-head self and cross-attention
- **FeedForwardBlock**: Position-wise feed-forward network
- **LayerNormalization**: Custom layer normalization
- **ResidualConnection**: Residual connections with dropout

### Architecture
- **Encoder**: 6 encoder layers with self-attention
- **Decoder**: 6 decoder layers with self and cross-attention
- **ProjectionLayer**: Final linear layer for vocabulary prediction

### Default Hyperparameters
- Model dimension: 128
- Number of layers: 6
- Number of attention heads: 8
- Feed-forward dimension: 2048
- Dropout rate: 0.1

## Evaluation Metrics

The model tracks several metrics during validation:
- Word-level accuracy
- Character Error Rate (CER)
- Word Error Rate (WER)
- BLEU Score

## Dataset

Uses the OPUS Books dataset for English-Swedish translation:
- Training: 90% of the dataset
- Validation: 10% of the dataset
- Tokenizers are trained on the dataset vocabulary
- Special tokens: [SOS], [EOS], [PAD], [UNK]

## Implementation Details

- Custom causal masking for decoder self-attention
- Xavier uniform parameter initialization
- Greedy decoding for inference
- Gradient clipping and learning rate scheduling available
- Automatic tokenizer generation and caching

## Files Description

- `model.py`: Complete Transformer implementation with all building blocks
- `train.py`: Training loop, validation, and inference functions
- `dataset.py`: Bilingual dataset class with proper masking
- `config.py`: Centralized configuration management

## License

This project is for educational purposes and implements the Transformer architecture as described in "Attention Is All You Need" by Vaswani et al. 