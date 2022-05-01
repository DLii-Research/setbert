/usr/bin/env python3 ./training/dnabert_pretrain.py \
    --length 150 \
    --kmer 3 \
    --embed-dim 128 \
    --stack 8 \
    --num-heads 4 \
    --pre-layernorm true \
    --batches-per-epoch 100 \
    --val-batches-per-epoch 16 \
    --data-augment true \
    --data-balance false \
    --data-artifact 'deep-learning-dna/dnasamples:latest' \
    --epochs 500 \
    --batch-size 512 \
    --mask-ratio 0.15 \
    --optimizer nadam \
    --lr 4e-4 \
    --init-lr 0.0 \
    --warmup-steps 10000