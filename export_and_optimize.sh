python export_and_optimize.py \
  --batch_size 32 \
  --max_seq_len 128 \
  --n_prompt_tokens 50 \
  --prompt_embed_dim 1024 \
  --cat_or_add "add" \
  --exported_model_name 'model' \
  --optimized_model_name 'optimized_model'