# NLP Text Summarization with BART

This project demonstrates fine-tuning a BART-base model for text summarization on the BillSum dataset, which contains US Congressional and California state bills along with their summaries.

## Project Overview

The notebook implements:

1. Data loading and preprocessing of the BillSum dataset
2. Fine-tuning a pre-trained BART-base model for summarization
3. Generating summaries on test data
4. Evaluating performance using ROUGE metrics

## Key Features

- Uses the `ainize/bart-base-cnn` pre-trained model
- Implements sequence-to-sequence training with Hugging Face Transformers
- Evaluates summarization quality with ROUGE scores
- Handles long document summarization (up to 1024 tokens)

## Results

The model achieves the following ROUGE scores on the test set:

| Metric     | Score  |
|------------|--------|
| ROUGE-1    | 0.334  |
| ROUGE-2    | 0.188  |
| ROUGE-L    | 0.224  |
| ROUGE-Lsum | 0.286  |

## Requirements

- Python 3.6+
- PyTorch
- Transformers library
- Datasets library
- Evaluate library
- Rouge-score

## Installation

```bash
pip install torch transformers datasets evaluate rouge-score 


