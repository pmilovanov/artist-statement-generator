#!/usr/bin/env bash

python3 /app/src/artstat/models/onehot.py sample --vocab_file=vocab_lower.txt \
    --vocab_is_lowercase=True \
    --seqlen=32 \
    --vocab_size=10000 \
    --model_file=/app/notebooks/checkpoints/m2/winner.hdf5 \
    --num_words_to_sample=1000 \
    --init_text="Mary had a little simulacrum, its exterior attributes white and cold. And everywhere that Mary went
    the simulacrum would also go."

# weights.lstm256.batch64.glove300.sample5.vocab10000.default.hdf5 \