import torch
import torch.nn as nn
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, accuracy_score
from transformers import GPT2Tokenizer
from utils import hinge_loss


class SST2Metric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.label_map = {
            tokenizer.encode('negative', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('positive', add_special_tokens=False)[0]: 1,  # positive
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()

        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class YelpPMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.label_map = {
            tokenizer.encode('negative', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('positive', add_special_tokens=False)[0]: 1,  # positive
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class AGNewsMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.label_map = {
            tokenizer.encode('world', add_special_tokens=False)[0]: 0,
            tokenizer.encode('sports', add_special_tokens=False)[0]: 1,
            tokenizer.encode('business', add_special_tokens=False)[0]: 2,
            tokenizer.encode('tech', add_special_tokens=False)[0]: 3,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class DBPediaMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.label_map = {
            tokenizer.encode('company', add_special_tokens=False)[0]: 0,
            tokenizer.encode('education', add_special_tokens=False)[0]: 1,
            tokenizer.encode('artist', add_special_tokens=False)[0]: 2,
            tokenizer.encode('athlete', add_special_tokens=False)[0]: 3,
            tokenizer.encode('office', add_special_tokens=False)[0]: 4,
            tokenizer.encode('transportation', add_special_tokens=False)[0]: 5,
            tokenizer.encode('building', add_special_tokens=False)[0]: 6,
            tokenizer.encode('natural', add_special_tokens=False)[0]: 7,
            tokenizer.encode('village', add_special_tokens=False)[0]: 8,
            tokenizer.encode('animal', add_special_tokens=False)[0]: 9,
            tokenizer.encode('plant', add_special_tokens=False)[0]: 10,
            tokenizer.encode('album', add_special_tokens=False)[0]: 11,
            tokenizer.encode('film', add_special_tokens=False)[0]: 12,
            tokenizer.encode('written', add_special_tokens=False)[0]: 13,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class MRPCMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.label_map = {
            tokenizer.encode('No', add_special_tokens=False)[0]: 0,  # not dumplicate
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 1,  # dumplicate
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        f1 = f1_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'f1': f1,
                'hinge': hinge_loss,
                'ce': ce_loss}


class MNLIMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.label_map = {
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 0,
            tokenizer.encode('Maybe', add_special_tokens=False)[0]: 1,
            tokenizer.encode('No', add_special_tokens=False)[0]: 2,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        target = target.cpu().numpy().tolist()
        for t in target:
            self._target.append(self.label_map[t])

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index].argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        if reset:
            self._target = []
            self._pred = []
        return {'acc': acc}


class RTEMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.label_map = {
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 0,
            tokenizer.encode('No', add_special_tokens=False)[0]: 1,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}


class SNLIMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.label_map = {
            tokenizer.encode('Yes', add_special_tokens=False)[0]: 0,
            tokenizer.encode('Maybe', add_special_tokens=False)[0]: 1,
            tokenizer.encode('No', add_special_tokens=False)[0]: 2,
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target == key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}