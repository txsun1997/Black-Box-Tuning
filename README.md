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
| Train       | 100           |
| Dev         | 96.88         |
| Test        | 88.3         |

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

In addition, black-box tuning also supports parallel evaluation. That is, you can evaluation a population of solutions in parallel by putting them into a single large batch. For example,

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

