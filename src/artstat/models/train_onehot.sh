#!/usr/bin/env bash

python3 /app/src/artstat/models/onehot.py train \
--glove_dims=300 \
--glove_file=/data/shared/glove/glove.6B.300d.txt \
--vocab_is_lowercase=True \
--vocab_file=vocab_lower.txt \
--vocab_size=10000 \
--seqlen=64 \
--sample_size=10 \
--batch_size=128 \
--learning_rate_decay_period=5 \
--learning_rate_decay_rate=0.9 \
--learning_rate_initial=0.003 \
--dropout_rate=0.01 \
--dense_layers=5 \
--dense_size=256 \
--lstm_size=256 \
--checkpoint_dir=/app/notebooks/checkpoints/m2 \
--training_data_dir=/data/local/artstat/train \
--num_epochs=100 \
--starting_epoch=0 \
--epochs_per_dataset=32 \
--starting_model_file=/app/notebooks/checkpoints/m2/1.hdf5

#weights.lstm256.batch128.glove300.sample10.vocab10000.default.hdf5
