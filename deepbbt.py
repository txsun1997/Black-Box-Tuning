import os
import copy
import time
import random

import torch
import fitlog
import argparse
import numpy as np
import cma
from fastNLP import cache_results, Tester, DataSet
from transformers import RobertaConfig, RobertaTokenizer
from deep_modeling_roberta import RobertaForMaskedLM
from dataloader import SST2Loader, AGNewsLoader, YelpPLoader, DBPediaLoader, RTELoader, MRPCLoader, SNLILoader
from metrics import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
from utils import hinge_loss
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default='sst2', type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--budget", default=8000, type=int)
parser.add_argument("--popsize", default=20, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
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
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every
# if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
#     args.cat_or_add = 'cat'
cat_or_add = args.cat_or_add

# fixed hyper-params
if cat_or_add == 'add':
    init_prompt_path = None
else:
    init_prompt_path = './nli_base_prompt.pt'

model_name = 'roberta-large'
# bound = math.sqrt(intrinsic_dim)
# bound = math.pow(intrinsic_dim, 0.75)
bound = 5

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

save_path = 'deep_results/{}_results/D_{}_d_{}_data_{}_{}_range_{}_loss_{}_budget_{}_seed_{}_{}_{}'.format(
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
    random_proj,
)
print('Results will be saved in {}'.format(save_path))

if os.path.exists(save_path):
    print('Experiment already run.')
    exit()

args.save_path = save_path
args.bound = bound

log_dir = './deeplogs'
fitlog.set_log_dir(log_dir)
fitlog.commit(__file__, fit_msg=save_path)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class LMForwardAPI:
    def __init__(self, model_name='roberta-large', n_prompt_tokens=50, task_name='sst2', save_path='./results',
                 loss_type='hinge'):
        self.config = RobertaConfig.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(
            model_name,
            config=self.config,
        )
        self.model.roberta.encoder.best_prefix = [
            torch.zeros(n_prompt_tokens, self.config.hidden_size, device='cuda:0')
            for _ in range(self.config.num_hidden_layers)
        ]
        self.model.roberta.encoder.n_prompt_tokens = n_prompt_tokens
        self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        self.init_prompt = None
        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.ModuleList([torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False) for _ in range(self.config.num_hidden_layers)])
        if random_proj == 'normal':
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, 0.0, 1.0 / intrinsic_dim)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_dev_loss = float('inf')
        self.best_prompt = None
        self.num_call = 0
        self.save_path = save_path
        self.print_every = print_every
        self.eval_every = eval_every
        self.loss_type = loss_type
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        if task_name == 'sst2':
            self.metric = SST2Metric(target='labels', pred='logits')
            self.metric_key = 'acc'
            self.metric_name = 'SST2Metric'
        elif task_name == 'agnews':
            self.metric = AGNewsMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
            self.metric_name = 'AGNewsMetric'
        elif task_name == 'yelpp':
            self.metric = YelpPMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
            self.metric_name = 'YelpPMetric'
        elif task_name == 'dbpedia':
            self.metric = DBPediaMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
            self.metric_name = 'DBPediaMetric'
        elif task_name == 'rte':
            self.metric = RTEMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
            self.metric_name = 'RTEMetric'
        elif task_name == 'mrpc':
            self.metric = MRPCMetric(target='labels', pred='logits')
            self.metric_key = 'f1'
            self.metric_name = 'MRPCMetric'
        elif task_name == 'snli':
            self.metric = SNLIMetric(target='labels', pred='logits')
            self.metric_key = 'acc'
            self.metric_name = 'SNLIMetric'
        else:
            raise NotImplementedError
        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduce='sum')

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduce='sum').item() / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target).item() / len(target)
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf


    def eval(self, prompt_embedding=None, layer_id=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            self.model.roberta.encoder.best_prefix = self.best_prompt
        else:
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear[layer_id](prompt_embedding)  # Az

        if self.init_prompt is not None:
            prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0


        if isinstance(test_data, DataSet):
            self.model.roberta.encoder.layer_id_to_replace = -1
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric, batch_size=batch_size,
                                 num_workers=4, device=device, use_tqdm=True)
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            self.model.roberta.encoder.layer_id_to_replace = layer_id
            self.model.roberta.encoder.prefix = prompt_embedding
            for k, v in train_data.items():
                train_data[k] = v.to(device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    mask_pos=train_data['mask_pos'],
                )['logits']

            loss, perf = self.calc_metric(logits, train_data['labels'])
            fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            fitlog.add_metric(perf, name='train_acc', step=self.num_call)

            if perf > self.best_train_perf:
                self.best_train_perf = perf
                fitlog.add_best_metric(self.best_train_perf, name='train_acc')

            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
                    fout.write('{}\t{}\t{}\n'.format(self.num_call, loss, perf))

            if self.num_call % self.print_every == 0:
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))

            if self.num_call % self.eval_every == 0:
                self.model.roberta.encoder.layer_id_to_replace = -1
                print('********* Evaluated on dev set *********')
                for k, v in dev_data.items():
                    dev_data[k] = v.to(device)
                with torch.no_grad():
                    logits = self.model(
                        input_ids=dev_data['input_ids'],
                        attention_mask=dev_data['attention_mask'],
                        mask_pos=dev_data['mask_pos'],
                    )['logits']

                dev_loss, dev_perf = self.calc_metric(logits, dev_data['labels'])
                fitlog.add_metric(dev_perf, name='dev_acc', step=self.num_call)
                if dev_perf > self.best_dev_perf:
                    self.best_dev_perf = dev_perf
                    fitlog.add_best_metric(self.best_dev_perf, name='dev_acc')
                if dev_loss <= self.best_dev_loss:
                    self.best_dev_loss = dev_loss
                    self.best_prompt = self.model.roberta.encoder.best_prefix
                if self.save_path is not None:
                    with open(os.path.join(self.save_path, 'dev_loss.txt'), 'a') as fout:
                        fout.write('{}\t{}\t{}\n'.format(self.num_call, dev_loss, dev_perf))
                print('Dev loss: {}. Dev perf: {}. Best dev loss: {}'.format(
                    round(float(dev_loss), 4),
                    round(float(dev_perf), 4),
                    round(float(self.best_dev_loss), 4)))
                print('********* Done *********')
            return loss


tokenizer = RobertaTokenizer.from_pretrained(model_name)
cache_fn = f"caches/data_{task_name}_{n_prompt_tokens}_{seed}.pt"
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
    if args.cat_or_add == 'cat':
        data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=0).my_load(splits)
    else:
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

train_data = {
    'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
    'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
    'mask_pos': torch.tensor(train_data['mask_pos'].get(list(range(len(train_data))))),
    'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
}

dev_data = {
    'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
    'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
    'mask_pos': torch.tensor(dev_data['mask_pos'].get(list(range(len(dev_data))))),
    'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
}

model_forward_api = LMForwardAPI(
    model_name=model_name,
    n_prompt_tokens=n_prompt_tokens,
    task_name=task_name,
    save_path=save_path,
    loss_type=loss_type,
)

cma_opts = {
    'seed': seed,
    'popsize': popsize,
    'bounds': [-1 * bound, 1 * bound],
    'maxiter': budget // (popsize * model_forward_api.config.num_hidden_layers),
    'verbose': -1,
}
es_list = [
    cma.CMAEvolutionStrategy(intrinsic_dim * [0], 0.5, inopts=cma_opts)
    for _ in range(model_forward_api.config.num_hidden_layers)
]

for _ in range(budget // (int(popsize) * model_forward_api.config.num_hidden_layers)):
    for i, es in enumerate(es_list):
        solutions = es.ask()
        fitnesses = [model_forward_api.eval(x, i) for x in solutions]
        es.tell(solutions, fitnesses)
        model_forward_api.model.roberta.encoder.best_prefix[i] = model_forward_api.linear[i](torch.tensor(es.result.xbest).type(torch.float32))  # set best cv

print(model_forward_api.model.roberta.encoder.best_prefix)
print('Evaluate on test data...')
test_acc = model_forward_api.eval(test_data=test_data)
print('Test acc: {}'.format(round(test_acc, 4)))
fitlog.finish()
