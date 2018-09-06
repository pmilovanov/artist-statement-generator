#!/usr/bin/env bash

python3 /app/src/artstat/models/onehot.py sample --vocab_file=vocab_lower.txt \
    --vocab_is_lowercase=True \
    --seqlen=64 \
    --vocab_size=10000 \
    --model_file=/app/notebooks/checkpoints/m2/weights.lstm1024.batch128.glove300.sample10.vocab10000.default.hdf5 \
    --num_words_to_sample=100 \
    --init_text="Mary had a little lamb"

# weights.lstm256.batch64.glove300.sample5.vocab10000.default.hdf5 \