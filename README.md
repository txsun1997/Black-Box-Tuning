# Black-Box-Tuning
Source code for paper "[Black-Box Tuning for Language-Model-as-a-Service](https://arxiv.org/abs/2201.03514)".

## Prepare your environment

The implementation of Black-Box Tuning is quite simple, you can check our code and easily implement it in your own environment. Or you can create a new environment to run our implementation, which is based on `pycma`, `Transformers` and `FastNLP`. Optionally, you can use `fitlog` to monitor experimental results. You can uncomment the fitlog-related lines in our code to use it.

```bash
conda create --name bbt python=3.8
conda activate bbt
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install cma
pip install sklearn
git clone https://github.com/txsun1997/Black-Box-Tuning
cd Black-Box-Tuning
```

## Optimize your prompt without gradients

Now you can run Black-Box Tuning with `run.sh`:

```bash
bash run.sh
```

Results will be saved in a directory named `results/`. In general, you will obtain the following results in ~8 minutes:

| SST-2 split | Best Accuracy |
| ----------- | ------------- |
| Train       | 100 %         |
| Dev         | 96.88 %       |
| Test        | 88.3 %        |

To reproduce other experiments in our paper, change the arguments of `bbt.py`, for example, 

```bash
python bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "hinge" \
  --cat_or_add "add" \
  --budget 5000 \
  --print_every 50 \
  --eval_every 100
```

In addition, black-box tuning also supports parallel evaluation. That is, you can evaluate a population of solutions in parallel by putting them into a single large batch. For example,

```bash
python bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "hinge" \
  --cat_or_add "add" \
  --budget 300 \
  --print_every 10 \
  --eval_every 20 \
  --parallel
```
## Inference Optimization
You can accelerate inference with Microsoft Onnxruntime. 
We provided an end-to-end inference optimization solution. 
Only one line of code is needed for ~2x inference speed.

To export a bbt model based on PyTorch to an Onnx model, 
you can run `export_and_optimize.py` with all arguments set to default to get a demo onnx model.
```bash
python export_and_optimize.py
```
Two models will be saved to `./onnx_models/`, namely exported (not accelerated) and optimized model.
Then you can modify `run.sh`. 
By setting parameter `inference_framework` to `'ort'` and `onnx_model_path` to `<Your model path>`,
a faster (but a little less accurate) version of BBT is ready.

To add some flexibility to model optimization, we provided some options in `export_and_optimize.py`.
You can adjust these arguments in `export_and_optimize.sh`. Here is an example.
```bash
python export_and_optimize.py \
  --batch_size 32 \
  --max_seq_len 128 \
  --n_prompt_tokens 50 \
  --prompt_embed_dim 1024 \
  --exported_model_name 'model' \
  --optimized_model_name 'optimized_model'
```
You can get the following results in 4.4 ± 0.1 minutes, 
compared to pytorch version of bbt whose training time is 8.8 ± 0.15 minutes (depends on hardware settings)

| SST-2 split | Best Accuracy   |
| ----------- | --------------- |
| Train       | 100 (no drop) % |
| Dev         | 96.88 (no drop)%|
| Test        | 86.7 (-1.6) %   |

## Cite

If you find this work helpful, please cite:

```bibtex
@article{sun2022bbt,
  title={Black-Box Tuning for Language-Model-as-as-Service}, 
  author={Tianxiang Sun and Yunfan Shao and Hong Qian and Xuanjing Huang and Xipeng Qiu},
  journal={arXiv preprint arXiv:2201.03514},
  year={2022}
}
```

