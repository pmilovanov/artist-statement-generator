#!/usr/bin/env bash

python3 /app/src/artstat/models/onehot.py train \
--glove_dims=300 \
--glove_file=/data/shared/glove/glove.6B.300d.txt \
--vocab_is_lowercase=True \
--vocab_file=vocab_lower.txt \
--vocab_size=10000 \
--seqlen=32 \
--sample_size=5 \
--batch_size=32 \
--learning_rate_decay_period=10 \
--learning_rate_decay_rate=0.7 \
--learning_rate_initial=0.01 \
--learning_rate_floor=0.00005 \
--dropout_rate=0.1 \
--dense_layers=6 \
--dense_size=300 \
--lstm_size=128 \
--lstm_layers=1 \
--checkpoint_dir=/app/notebooks/checkpoints/m2 \
--training_data_dir=/data/local/artstat/train \
--num_epochs=1000 \
--starting_epoch=0 \
--epochs_per_dataset=4
##--starting_model_file=/app/notebooks/checkpoints/m2/weights.lstm1024.batch128.glove300.sample10.vocab10000.default.hdf5

#--starting_model_file=/app/notebooks/checkpoints/m2/weights.lstm1024.batch128.glove300.sample10.vocab10000.default.hdf5
#--training_max_files=3000 \

#lr=0.01, period=10, rate=0.7, dropout=0.1, floor=0.00005

#after epoch 49, lr=0.0024; loss=1.1636, acc=0.8712