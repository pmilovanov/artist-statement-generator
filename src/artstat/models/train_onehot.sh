#!/usr/bin/env bash

python3 /app/src/artstat/models/onehot.py train \
--glove_dims=300 \
--glove_file=/data/shared/glove/glove.6B.300d.txt \
--vocab_is_lowercase=True \
--vocab_file=vocab_lower.txt \
--vocab_size=10000 \
--seqlen=64 \
--sample_size=5 \
--batch_size=64 \
--learning_rate_decay_period=1 \
--learning_rate_decay_rate=0.9 \
--learning_rate_initial=0.001 \
--dropout_rate=0.01 \
--dense_layers=10 \
--dense_size=128 \
--lstm_size=256 \
--checkpoint_dir=/app/notebooks/checkpoints/m2 \
--training_data_dir=/data/local/artstat/train \
--num_epochs=100 \
--starting_epoch=8 \
--epochs_per_dataset=128 \
--starting_model_file=/app/notebooks/checkpoints/m2/weights.lstm256.batch64.glove300.sample5.vocab10000.default.hdf5
