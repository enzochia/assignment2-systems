 uv run --active -m cs336_systems.run_scripts.benchmark \
    --context_length 256 \
    --batch_size 64 \
    --vocab_size 10000 \
    --d_model 512 \
    --d_ff 1344 \
    --num_heads 16 \
    --num_layers 4 \
    --warmup_iters 5 \
    --benchmarking_iters 10 \
    --rope_theta 10000




