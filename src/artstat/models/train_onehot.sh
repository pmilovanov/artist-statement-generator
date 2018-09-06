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
--learning_rate_decay_period=1 \
--learning_rate_decay_rate=0.99 \
--learning_rate_initial=0.01 \
--dropout_rate=0.00 \
--dense_layers=4 \
--dense_size=300 \
--lstm_size=1024 \
--lstm_layers=1 \
--checkpoint_dir=/app/notebooks/checkpoints/m2 \
--training_data_dir=/data/local/artstat/train \
--num_epochs=100 \
--starting_epoch=6 \
--epochs_per_dataset=48
#--starting_model_file=/app/notebooks/checkpoints/m2/weights.lstm1024.batch128.glove300.sample10.vocab10000.default.hdf5
#--training_max_files=3000 \