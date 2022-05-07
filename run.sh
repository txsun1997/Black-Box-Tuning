for task in "mrpc" "rte" "snli" "agnews" "sst2" "dbpedia" "yelpp"
do
CUDA_VISIBLE_DEVICES=0 python deepbbt.py \
  --task_name $task \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 100 \
  --loss_type "hinge" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100 \
  --inference_framework 'ort' \
  --onnx_model_path './onnx_models/deep_optimized_model.onnx'
done