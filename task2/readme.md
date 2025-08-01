# Bigram Transformer with RoPE and Mixture-of-Experts (MoE)

Этот проект демонстрирует минимальную реализацию **Transformer языковой модели**, в которой:
- Используются **Rotary Positional Embeddings (RoPE)** для позиционной информации внутри Self-Attention.
- Реализован **Mixture-of-Experts (MoE)** слой вместо обычного FeedForward слоя.
- Используется **RMSNorm** вместо LayerNorm.
- Можно обучить и сгенерировать текст на основе вашего корпуса.


## Основные гиперпараметры

- `batch_size` = 16 — размер батча.
- `block_size` = 32 — длина контекста.
- `n_embd` = 64 — размер эмбеддингов.
- `n_head` = 4 — число attention голов.
- `n_layer` = 4 — количество блоков Transformer.
- `dropout` = 0.5 — dropout.
- `learning_rate` = 1e-3 — скорость обучения.
- `max_iters` = 5000 — количество итераций тренировки.

## Архитектура

**Модель состоит из:**
- Embedding таблицы для токенов.
- Стэка `Block`-ов: каждый блок — это Multi-Head Attention с RoPE + MoE FeedForward.
- RMSNorm для нормализации.
- Linear layer для предсказания вероятности следующего токена.

**Особенности:**
- `Head` — одна голова Self-Attention с применением RoPE.
- `MultiHeadAttention` — несколько голов в параллель.
- `MoEFeedForward` — слой Mixture-of-Experts: несколько экспертов и роутер.
- `RMSNorm` — нормализация по RMS.
- `BigramLanguageModel` — сама модель, объединяющая все блоки.



