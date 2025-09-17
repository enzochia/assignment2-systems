 uv run --active -m cs336_systems.run_scripts.naive_ddp \
    --data_path ../../assignment_1/CS336-assignment1-basics/data/ts/encoded/ \
    --context_length 2048 \
    --batch_size 4 \
    --vocab_size 10000 \
    --d_model 768 \
    --d_ff 3072 \
    --num_heads 12 \
    --num_layers 12 \
    --rope_theta 1e4 \
    --activation_function SwiGLU \
    --target_token_count 512000000 \
    --cosine_cycle_iters 24000 \
    --max_learning_rate 1e-3 \
    --min_learning_rate 1e-6 \
    --grad_clip_max_l2_norm 1 \
    --optim_weight_decay 1e-2 \
    --optim_eps 1e-8 \
    --adamw_betas 0.9 0.95 \
    --optim_lr 1e-3 \
    --log_every 1 \
    --eval_every 5000 \
    --eval_iters 10 \
    --save_checkpoint_every 30000 \
    --checkpoint_path data/ckpt/pretrained/ \
    --init_from scratch \
    --init_from_path data/ckpt/pretrained/iter_24000/ \
    --sampling_mode random 
    # --wandb_logging \
    # --wandb_project cs336-hw1-enzojia \
    # --wandb_run_name test-001
   #  --warmup_iters 100 \
   #  --cosine_cycle_iters 800 \
   #  --init_from pretrained \
   #  --init_from_path data/ckpt/pretrained/iter_24000/ \




