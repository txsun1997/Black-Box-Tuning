python export_and_optimize.py \
  --batch_size 32 \
  --max_seq_len 128 \
  --n_prompt_tokens 50 \
  --prompt_embed_dim 1024 \
  --cat_or_add "add" \
  --exported_model_name 'deep_model' \
  --optimized_model_name 'deep_optimized_model' \
  --deep