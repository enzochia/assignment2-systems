 uv run --active -m cs336_systems.run_scripts.benchmark \
    --context_length 512 \
    --batch_size 1 \
    --vocab_size 10000 \
    --d_model 1024 \
    --d_ff 4096 \
    --num_layers 24 \
    --num_heads 16 \
    --warmup_iters 5 \
    --benchmarking_iters 10 \
    --rope_theta 10000 \
    --benchmark_memory \
    --benchmark_memory_path memory_snapshots/memory_snapshot.pickle
   #  --use_mixed_precision \
   #  --precision fp16 \





