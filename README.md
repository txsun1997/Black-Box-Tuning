# Black-Box-Tuning
Source code for paper "[Black-Box Tuning for Language-Model-as-a-Service](https://arxiv.org/abs/2201.03514)".

> Being busy recently, the code in this repo and this tutorial will be very brief. Please let me know if you find any issues.

## Prepare your environment

The implementation of Black-Box Tuning is quite simple, you can check our code and easily implement it in your own environment. Or you can create a new environment to run our implementation, which is based on `Nevergrad`, `Transformers` and `FastNLP`. Optionally, we use `fitlog` to monitor experimental results. You can uncomment the fitlog-related lines in our code to use it.

```bash
conda create --name bbt python=3.8
conda activate bbt
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install nevergrad
pip install sklearn
git clone https://github.com/txsun1997/Black-Box-Tuning
cd Black-Box-Tuning
```

## Optimize your prompt without gradients

Now you can run Black-Box Tuning with `run.sh`:

```bash
bash run.sh
```

Results will be saved in a directory named `results/`. To reproduce other experiments in our paper, change the arguments of `bbt.py`, for example, 

```bash
python bbt.py --task_name "agnews" --n_prompt_tokens 50 --intrinsic_dim 500 --k_shot 16 --device "cuda:0" --seed 42 --loss_type "hinge" --cat_or_add "add" --budget 8000
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

