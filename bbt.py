import os
import copy
import random

import torch
# import fitlog
import argparse
import numpy as np
import nevergrad as ng
from fastNLP import cache_results, Tester, DataSet
from transformers import RobertaConfig, RobertaTokenizer
from modeling_roberta import RobertaForMaskedLM
from dataloader import SST2Loader, AGNewsLoader, YelpPLoader, DBPediaLoader, RTELoader, MRPCLoader, SNLILoader
# from metrics import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
from metrics_fast import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric


parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default='sst2', type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=200, type=int)
parser.add_argument("--budget", default=8000, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--alg", default='CMA', type=str)
parser.add_argument("--random_proj", default='he', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--loss_type", default='hinge', type=str)
parser.add_argument("--cat_or_add", default='add', type=str)
args = parser.parse_args()

# below are free hyper-params
task_name = args.task_name
n_prompt_tokens = args.n_prompt_tokens
intrinsic_dim = args.intrinsic_dim
k_shot = args.k_shot
batch_size = args.batch_size
budget = args.budget
device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
# if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
#     args.cat_or_add = 'cat'
cat_or_add = args.cat_or_add

# fixed hyper-params
if cat_or_add == 'add':
    init_prompt_path = None
else:
    init_prompt_path = 'YOUR_PROMPT_PATH'

model_name = 'roberta-large'
# bound = math.sqrt(intrinsic_dim)
# bound = math.pow(intrinsic_dim, 0.75)
bound = 5
eval_every = 100

if task_name in ['sst2', 'yelpp', 'rte', 'mrpc']:
    num_labels = 2
elif task_name in ['snli']:
    num_labels = 3
elif task_name in ['agnews']:
    num_labels = 4
elif task_name in ['dbpedia']:
    num_labels = 14
else:
    raise ValueError

save_path = 'results/{}_results/D_{}_d_{}_data_{}_{}_range_{}_loss_{}_budget_{}_seed_{}_{}_{}'.format(
    task_name,
    n_prompt_tokens * 1024,
    intrinsic_dim,
    k_shot * num_labels,
    alg,
    bound,
    loss_type,
    budget,
    seed,
    cat_or_add,
    random_proj
)
print('Results will be saved in {}'.format(save_path))

if os.path.exists(save_path):
    print('Experiment already run.')
    exit()

args.save_path = save_path
args.bound = bound

# log_dir = './logs'
# fitlog.set_log_dir(log_dir)
# fitlog.commit(__file__, fit_msg=save_path)
# fitlog.add_hyper(args)
# fitlog.add_hyper_in_file(__file__)


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class LMForwardAPI:
    def __init__(self, model_name='roberta-large', n_prompt_tokens=50, task_name='sst2', save_path='./results',
                 loss_type='hinge', init_prompt_path=None):
        self.config = RobertaConfig.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name, config=self.config,
                                                        n_prompt_tokens=n_prompt_tokens)
        self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        if cat_or_add == 'cat':
            self.model.set_concat_prompt(True)
            if init_prompt_path is not None:
                print('Initialize prompt embedding from {}'.format(init_prompt_path))
                self.init_prompt = torch.load(init_prompt_path).weight.cpu().reshape(-1)
            else:
                print('Initial prompt embedding not found. Initialize to zero embedding.')
                self.init_prompt = torch.zeros(n_prompt_tokens * self.config.hidden_size)
            print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        else:
            self.model.set_concat_prompt(False)
            self.init_prompt = None
        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False)
        if random_proj == 'normal':
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, 0.0, 1.0 / intrinsic_dim)
        self.task_name = task_name
        if task_name == 'sst2':
            self.metric = SST2Metric(target='labels', pred='logits')
            self.metric_key = 'acc'
        elif task_name == 'agnews':
            self.metric = AGNewsMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
        elif task_name == 'yelpp':
            self.metric = YelpPMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
        elif task_name == 'dbpedia':
            self.metric = DBPediaMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
        elif task_name == 'rte':
            self.metric = RTEMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
        elif task_name == 'mrpc':
            self.metric = MRPCMetric(target='labels', pred='logits')
            self.metric_key = 'f1'
        elif task_name == 'snli':
            self.metric = SNLIMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
        else:
            raise NotImplementedError
        self.tester = Tester(data=train_data, model=self.model, metrics=self.metric, batch_size=batch_size,
                             num_workers=4, device=device, verbose=1, use_tqdm=False)
        self.dev_tester = Tester(data=dev_data, model=self.model, metrics=self.metric, batch_size=batch_size,
                                 num_workers=4, device=device, verbose=1, use_tqdm=False)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        self.save_path = save_path
        self.eval_every = eval_every
        self.loss_type = loss_type
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

    def get_tokenizer(self):
        return self.tokenizer

    def zero_shot_eval(self):
        self.model.set_prompt_embedding(None)
        print('Evaluating on training data...')
        self.tester.test()
        print('Evaluating on dev data...')
        self.dev_tester.test()

    def eval(self, prompt_embedding, test_data=None):
        self.num_call += 1
        tmp_prompt = copy.deepcopy(prompt_embedding)
        prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
        prompt_embedding = self.linear(prompt_embedding)  # Az
        if self.init_prompt is not None:
            prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
        # print(prompt_embedding.view(n_prompt_tokens, -1))
        self.model.set_prompt_embedding(prompt_embedding)

        if self.task_name == 'sst2':
            metric_name = 'SST2Metric'
        elif self.task_name == 'agnews':
            metric_name = 'AGNewsMetric'
        elif self.task_name == 'yelpp':
            metric_name = 'YelpPMetric'
        elif self.task_name == 'dbpedia':
            metric_name = 'DBPediaMetric'
        elif self.task_name == 'rte':
            metric_name = 'RTEMetric'
        elif self.task_name == 'mrpc':
            metric_name = 'MRPCMetric'
        elif self.task_name == 'snli':
            metric_name = 'SNLIMetric'
        else:
            raise NotImplementedError

        if test_data is not None:
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric, batch_size=batch_size,
                                 num_workers=4, device=device, verbose=1, use_tqdm=True)
            results = test_tester.test()
            test_acc = results[metric_name][self.metric_key]
            # fitlog.add_best_metric(test_acc, name='test_acc')
        else:
            results = self.tester.test()

            perf = -1.0 * results[metric_name][self.metric_key]  # -accuracy
            hinge_loss = results[metric_name]['hinge']
            ce_loss = results[metric_name]['ce']

            # fitlog.add_loss(hinge_loss, name='hinge', step=self.num_call)
            # fitlog.add_metric(-1.0 * perf, name='train_acc', step=self.num_call)

            if perf < self.best_train_perf:
                self.best_train_perf = perf
                # fitlog.add_best_metric(-1.0 * self.best_train_perf, name='train_acc')

            print(
                '[# API Calls {}] Hinge loss: {}. Cross entropy loss: {}. Current performance: {}. Best performance so far: {}'.format(
                    self.num_call,
                    hinge_loss,
                    ce_loss,
                    -1.0 * perf,
                    -1.0 * self.best_train_perf))

            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
                    fout.write('{}\t{}\n'.format(self.num_call, -1.0 * perf))

            if self.num_call % self.eval_every == 0:
                print('********* Evaluated on dev set *********')
                dev_results = self.dev_tester.test()
                dev_perf = dev_results[metric_name][self.metric_key]
                # fitlog.add_metric(dev_perf, name='dev_acc', step=self.num_call)
                if dev_perf > self.best_dev_perf:
                    self.best_dev_perf = dev_perf
                    # fitlog.add_best_metric(self.best_dev_perf, name='dev_acc')
                    self.best_prompt = copy.deepcopy(tmp_prompt)
                if self.save_path is not None:
                    with open(os.path.join(self.save_path, 'dev_acc.txt'), 'a') as fout:
                        fout.write('{}\t{}\n'.format(self.num_call, dev_perf))
                print('********* Done *********')
            if self.loss_type == 'perf':
                return perf
            elif self.loss_type == 'hinge':
                return hinge_loss
            elif self.loss_type == 'ce':
                return ce_loss
            else:
                raise ValueError


tokenizer = RobertaTokenizer.from_pretrained(model_name)
cache_fn = f"caches/data_{task_name}_{n_prompt_tokens}.pt"
DataLoader = {
    'sst2': SST2Loader,
    'agnews': AGNewsLoader,
    'yelpp': YelpPLoader,
    'dbpedia': DBPediaLoader,
    'rte': RTELoader,
    'mrpc': MRPCLoader,
    'snli': SNLILoader,
}


@cache_results(cache_fn, _refresh=False)
def get_data(task_name, tokenizer):
    if task_name in ['agnews', 'yelpp', 'dbpedia', 'snli']:
        splits = ['train', 'test']
    else:  # for datasets without test set, we use dev set
        splits = ['train', 'validation']
    data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens).my_load(splits)
    return data_bundle


def construct_true_few_shot_data(train_data, k_shot):
    train_label_count = {}
    dev_label_count = {}
    new_train_data = DataSet()
    new_dev_data = DataSet()
    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    for index in all_indices:
        label = train_data[index]['labels']
        if label < 0:
            continue

        if label not in train_label_count:
            train_label_count[label] = 0
        if label not in dev_label_count:
            dev_label_count[label] = 0

        if train_label_count[label] < k_shot:
            new_train_data.append(train_data[index])
            train_label_count[label] += 1
        elif dev_label_count[label] < k_shot:
            new_dev_data.append(train_data[index])
            dev_label_count[label] += 1
    new_train_data.set_input("input_ids", "attention_mask", "labels", "mask_pos")
    new_train_data.set_target("labels")
    new_dev_data.set_input("input_ids", "attention_mask", "labels", "mask_pos")
    new_dev_data.set_target("labels")
    return new_train_data, new_dev_data


data_bundle = get_data(task_name=task_name, tokenizer=tokenizer)
if task_name in ['agnews', 'yelpp', 'dbpedia', 'snli']:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('test')
else:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('validation')

train_data, dev_data = construct_true_few_shot_data(train_data, k_shot)
print('# of train data: {}'.format(len(train_data)))
print('Example:')
print(train_data[0])
print('\n# of dev data: {}'.format(len(dev_data)))
print('Example:')
print(dev_data[0])
print('\n# of test data: {}'.format(len(test_data)))
print('Example:')
print(test_data[0])

model_forward_api = LMForwardAPI(
    model_name=model_name,
    n_prompt_tokens=n_prompt_tokens,
    task_name=task_name,
    save_path=save_path,
    loss_type=loss_type,
    init_prompt_path=init_prompt_path
)

# print('********* Zero-shot Performance *********')
# model_forward_api.zero_shot_eval()
# model_forward_api.num_call = 0
# print('********* Done *********')

if bound > 0:
    parametrization = ng.p.Array(shape=(intrinsic_dim,)).set_bounds(lower=-1 * bound, upper=bound)
else:
    parametrization = ng.p.Array(shape=(intrinsic_dim,))

optim = ng.optimizers.registry[alg](parametrization=parametrization, budget=budget, num_workers=4)
for i in range(budget):
    x = optim.ask()
    y = model_forward_api.eval(*x.args)
    optim.tell(x, y)
print("Done.")
# recommendation = optim.recommend()
best_prompt = model_forward_api.best_prompt
test_acc = model_forward_api.eval(prompt_embedding=best_prompt, test_data=test_data)
# fitlog.finish()
