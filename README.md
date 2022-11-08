# Black-Box-Tuning for Language-Model-as-a-Service
## Updates

- 2022/10/14: Release the latest version of BBTv2, check out the [updated results](https://docs.google.com/spreadsheets/d/1FA9zMW613OoskI_fBXuZNNBVo7RFV5OvdZ0YHnA2j5Q/edit?usp=sharing). :mag:
- 2022/07/05: Release a [paper list](https://github.com/txsun1997/LMaaS-Papers) on LMaaS, check out other awesome papers! :bookmark_tabs:
- 2022/06/05: Support T5 and GPT-2 model. :clap:
- 2022/05/15: Support BERT and BART model. :clap:
- 2022/05/04: Release BBTv2, check out our [paper](https://arxiv.org/abs/2205.11200) and try it with `deepbbt.py` :tada:
- 2022/02/18: Support ONNX Runtime optimization (training speed is doubled!) :rocket:
- 2022/01/13: Release the first version of BBT, check out our [paper](https://arxiv.org/abs/2201.03514). :tada:

## Quick Links

- [Introduction](#introduction)
- [Prepare your environment](#prepare-your-environment)
- [Using BBT](#using-bbt)
- [Using BBTv2](#using-bbtv2)
- [Inference Optimization](#inference-optimization)
- [Citation](#citation)

## Introduction

Black-Box Tuning (BBT) is a gradient-free method to drive large language models (LLMs) for few-shot learning. It optimizes a sequence of soft prompt tokens prepended to the input of LLMs, without requiring gradients/back-propagation of the LLMs. Therefore, pre-trained general-purposed LLMs can be viewed as black-box models and deployed efficiently on some inference servers. In such a scenario, which we call Language-Model-as-a-Service (LMaaS), BBT can achieve comparable performance to full model tuning by only accessing model inference APIs. Generally, BBT can achieve considerable results on most language understanding datasets within 8k model forward passes.

More details are provided in our ICML paper [Black-Box Tuning for Language-Model-as-a-Service](https://arxiv.org/abs/2201.03514) and our EMNLP paper [BBTv2: Towards a Gradient-Free Future with Large Language Models](https://arxiv.org/abs/2205.11200).

> To help reproduce results reported in the paper, we also release a [Google Sheets](https://docs.google.com/spreadsheets/d/1FA9zMW613OoskI_fBXuZNNBVo7RFV5OvdZ0YHnA2j5Q/edit?usp=sharing) recording BBTv2 performance on each dataset using each random seed. Feel free to reach out to me if you cannot obtain similar results.

## Prepare your environment

The implementation of Black-Box Tuning is quite simple, you can check our code and easily implement it in your own environment. Or you can create a new environment to run our implementation based on `pycma`, `Transformers` and `FastNLP`. Optionally, you can use `fitlog` to monitor experimental results. You can uncomment the fitlog-related lines in our code to use it.

```bash
conda create --name bbt python=3.8
conda activate bbt
pip install transformers==4.1.1
pip install fastNLP==0.6.0
pip install datasets
pip install cma
pip install sklearn
git clone https://github.com/txsun1997/Black-Box-Tuning
cd Black-Box-Tuning
```

## Using BBT

Now you can run Black-Box Tuning with `run.sh`:

```bash
bash run.sh
```

In general, you will obtain the following results in ~13 minutes (tested on NVIDIA 3090 GPU):

| SST-2 split | Best Accuracy |
| ----------- | ------------- |
| Train       | 100 %         |
| Dev         | 96.88 %       |
| Test        | 90.48 %       |

To reproduce other experiments in our paper, change the arguments of `bbt.py`, for example, 

```bash
python bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100
```

To obtain similar results as reported in the original paper, we recommend using `--loss_type "hinge"` for sentence-pair tasks (i.e., MRPC, SNLI, and RTE) and using `--budget 20000` for DBPedia.

In addition, black-box tuning also supports parallel evaluation. That is, you can evaluate a population of solutions in parallel by putting them into a single large batch. For example,

```bash
python bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 300 \
  --print_every 10 \
  --eval_every 20 \
  --parallel
```
## Using BBTv2

BBTv2 is an improved version of BBT. Instead of optimizing the prompt merely in the input layer, BBTv2 adopts a divide-and-conquer algorithm to alternately optimize prompts in every layer (i.e., deep prompt). You can simply try BBTv2 using the following command,

```bash
python deepbbt.py \
  --model_name "roberta-large"\
  --task_name "agnews" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --random_proj "normal" \
  --sigma 0.2 \
  --alpha 0.2 \
  --popsize 20 \
  --bound 0 \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100
```

BBTv2 usually confers better results on many label classification tasks (e.g., DBPedia) and entailment tasks (e.g., MRPC, SNLI, RTE, etc.). Check our [Google Sheets](https://docs.google.com/spreadsheets/d/1FA9zMW613OoskI_fBXuZNNBVo7RFV5OvdZ0YHnA2j5Q/edit?usp=sharing) if you have problem reproducing the results of BBTv2.

## Inference Optimization

In contrast to training with gradient descent, BBT (and BBTv2) only requires model forward computation, and therefore can be significantly accelerated using [ONNX Runtime](https://onnxruntime.ai/) or NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt). 

Here we provide an implementation of inference optimization using ONNX Runtime. You can obtain ~2x speedup using only one line of code.

SDK `onnxruntime-gpu` is required for optimization. Installation of this package can be troublesome. And there may be some environment-specific errors or unexpected performance. But in real-world scenarios, this is a part of the black box on the server side.

The following code works well to configure the environment on an NVIDIA GeForce RTX 3090 GPU with Driver Version: 470.82.00 and CUDA Version: 11.4.
```bash
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install cma
pip install sklearn
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install onnx
pip install onnxruntime-gpu==1.10.0
pip install coloredlogs
pip install sympy
```

To export a BBT model based on `PyTorch` to an `ONNX` model, 
you can run `export_and_optimize.py` with all arguments set to default to get a demo onnx model.

```bash
python export_and_optimize.py
```
Two models will be saved to `./onnx_models/`, namely exported (not accelerated) and optimized model.
Then you can modify `run.sh`. 
By setting parameter `inference_framework` to `'ort'` and `onnx_model_path` to `<Your model path>`,
a faster version of BBT is ready. Here is an example.
```bash
python bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100 \
  --inference_framework 'ort' \
  --onnx_model_path './onnx_models/optimized_model.onnx'
```

To add some flexibility to model optimization, we provided some options in `export_and_optimize.py`.
You can adjust these arguments in `export_and_optimize.sh`. Here is an example.
```bash
python export_and_optimize.py \
  --batch_size 32 \
  --max_seq_len 128 \
  --n_prompt_tokens 50 \
  --prompt_embed_dim 1024 \
  --cat_or_add "add" \
  --exported_model_name 'model' \
  --optimized_model_name 'optimized_model'
```
Onnx models are static, but to cat or to add is a branch in the model. During building phase, unused nodes in the model graph are removed for better performance. So you have to build one for each mode.

You can get the following results in 4.3 ± 0.1 minutes, compared to pytorch version of BBT whose training time is 8.9 ± 0.15 minutes (depends on hardware settings)

You can get the following results by running BBT 100 times on sst2 with random seed set from 1 to 100.
Fp16 optimization does not hurt performance on all tasks.

| SST-2 split | Best Accuracy   |
| ----------- | --------------- |
| Test        | 88.0  %   |

## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{sun2022bbt,
  title={Black-Box Tuning for Language-Model-as-a-Service}, 
  author={Tianxiang Sun and Yunfan Shao and Hong Qian and Xuanjing Huang and Xipeng Qiu},
  booktitle = {Proceedings of {ICML}},
  year={2022}
}
```

```bibtex
@inproceedings{sun2022bbtv2,
  title={BBTv2: Towards a Gradient-Free Future with Large Language Models},
  author={Tianxiang Sun and Zhengfu He and Hong Qian and Yunhua Zhou and Xuanjing Huang and Xipeng Qiu},
  booktitle = {Proceedings of {EMNLP}},
  year={2022}
}
```

