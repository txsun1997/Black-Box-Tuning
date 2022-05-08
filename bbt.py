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
from modeling_roberta import RobertaForMaskedLM
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
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str,
    help='''Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
)
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    help='Path to your onnx model.'
)
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
parallel = args.parallel
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path

if inference_framework not in ['pt', 'ort']:
    raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
if inference_framework == 'ort':
    assert onnx_model_path is not None, 'Path to onnx model is required, got None instead.'
    assert os.path.exists(onnx_model_path), f'In valid onnx model path `{onnx_model_path}`'

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

save_path = 'results/{}_results/D_{}_d_{}_data_{}_{}_range_{}_loss_{}_budget_{}_seed_{}_{}_{}_{}_{}'.format(
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
    'parallel' if parallel else 'serial',
    inference_framework
)
print('Results will be saved in {}'.format(save_path))

if os.path.exists(save_path):
    print('Experiment already run.')
    exit()

args.save_path = save_path
args.bound = bound
args.bbt_version = 'bbt'

log_dir = './logs'
fitlog.set_log_dir(log_dir)
fitlog.commit(__file__, fit_msg=save_path)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class LMForwardAPI:
    def __init__(self, model_name='roberta-large', n_prompt_tokens=50, task_name='sst2', save_path='./results',
                 loss_type='hinge', init_prompt_path=None):
        self.config = RobertaConfig.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(
            model_name,
            config=self.config,
            n_prompt_tokens=n_prompt_tokens,
            inference_framework=inference_framework,
            onnx_model_path=onnx_model_path,
        )
        if inference_framework == 'ort':
            self.model.roberta = None
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
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
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

    def eval(self, prompt_embedding=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        if test_data is None:
            bsz = len(dev_data['input_ids'])  # batch size of dev data is the orignal batch size of training data
        else:
            bsz = batch_size  # for test data
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(z.reshape(n_prompt_tokens, -1).repeat(bsz, 1, 1))
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(train_data['input_ids'])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)

        if isinstance(test_data, DataSet):
            if prompt_embedding.shape[0] > bsz:
                raise ValueError('Provide a single prompt embedding for testing.')
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric, batch_size=batch_size,
                                 num_workers=4, device=device, use_tqdm=True)
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            for k, v in train_data.items():
                train_data[k] = v.to(device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    mask_pos=train_data['mask_pos'],
                )['logits']

            if parallel:  # we have multiple queries
                all_losses, all_perfs = [], []
                for i in range(len(logits) // bsz):
                    tmp_logits = logits[i * bsz:i * bsz + bsz]
                    tmp_target = train_data['labels'][i * bsz:i * bsz + bsz]
                    tmp_loss, tmp_perf = self.calc_metric(tmp_logits, tmp_target)
                    all_losses.append(tmp_loss)
                    all_perfs.append(tmp_perf)
                loss = min(all_losses)
                best_sol = all_losses.index(loss)  # argmin
                perf = all_perfs[best_sol]  # corresponding performance
                tmp_prompt = tmp_prompt[best_sol]  # numpy.ndarray
                prompt_embedding = pe_list[best_sol]  # to be prepended to the input
            else:  # single query
                loss, perf = self.calc_metric(logits, train_data['labels'])
            fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            fitlog.add_metric(perf, name='train_acc', step=self.num_call)

            if perf > self.best_train_perf:
                self.best_train_perf = perf
                fitlog.add_best_metric(self.best_train_perf, name='train_acc')

            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
                    fout.write('{}\t{}\n'.format(self.num_call, perf))

            if self.num_call % self.print_every == 0:
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))

            if self.num_call % self.eval_every == 0:
                print('********* Evaluated on dev set *********')
                if parallel:  # if we have multiple queries, use the one that achieves minimal loss
                    self.model.set_prompt_embedding(prompt_embedding)
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
                    self.best_prompt = copy.deepcopy(tmp_prompt)
                if self.save_path is not None:
                    with open(os.path.join(self.save_path, 'dev_acc.txt'), 'a') as fout:
                        fout.write('{}\t{}\n'.format(self.num_call, dev_loss))
                print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
                    round(float(dev_loss), 4),
                    round(float(dev_perf), 4),
                    round(float(self.best_dev_perf), 4)))
                print('********* Done *********')
            if parallel:
                return all_losses
            else:
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
    init_prompt_path=init_prompt_path
)

cma_opts = {
    'seed': seed,
    'popsize': popsize,
    'bounds': [-1 * bound, 1 * bound],
    'maxiter': budget if parallel else budget // popsize,
    'verbose': -1,
}
es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], 0.5, inopts=cma_opts)
print('Population Size: {}'.format(es.popsize))
print('{} Evaluation.'.format('Parallel' if parallel else 'Serial'))
if parallel:
    # expand training data to a larger batch for parallel evaluation
    train_data['input_ids'] = train_data['input_ids'].repeat(es.popsize, 1)
    train_data['attention_mask'] = train_data['attention_mask'].repeat(es.popsize, 1)
    train_data['mask_pos'] = train_data['mask_pos'].repeat(es.popsize)
    train_data['labels'] = train_data['labels'].repeat(es.popsize)

# opt = cma.CMAOptions()
start_time = time.time()
while not es.stop():
    solutions = es.ask()
    if parallel:
        fitnesses = model_forward_api.eval(solutions)
    else:
        fitnesses = [model_forward_api.eval(x) for x in solutions]
    es.tell(solutions, fitnesses)
    # es.logger.add()  # write data to disc to be plotted
    # es.disp()
end_time = time.time()
print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
print('Evaluate on test data...')
test_acc = model_forward_api.eval(test_data=test_data)
print('Test acc: {}'.format(round(test_acc, 4)))
fitlog.finish()
