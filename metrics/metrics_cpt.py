import torch
import torch.nn as nn
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, accuracy_score
from transformers import RobertaTokenizer
from utils import hinge_loss


class BasicMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduce='sum')
        self.margin = 2

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


class ChnSentMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('差', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('好', add_special_tokens=False)[0]: 1,  # positive
        }


class THUCNewsMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('体育', add_special_tokens=False)[0]: 0,
            tokenizer.encode('娱乐', add_special_tokens=False)[0]: 1,
            tokenizer.encode('房产', add_special_tokens=False)[0]: 2,
            tokenizer.encode('教育', add_special_tokens=False)[0]: 3,
            tokenizer.encode('时尚', add_special_tokens=False)[0]: 4,
            tokenizer.encode('政治', add_special_tokens=False)[0]: 5,
            tokenizer.encode('游戏', add_special_tokens=False)[0]: 6,
            tokenizer.encode('社会', add_special_tokens=False)[0]: 7,
            tokenizer.encode('科技', add_special_tokens=False)[0]: 8,
            tokenizer.encode('经济', add_special_tokens=False)[0]: 9,

        }

class LCQMCMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('矛盾', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('相似', add_special_tokens=False)[0]: 1,  # positive
        }


class CMNLIMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('矛盾', add_special_tokens=False)[0]: 0,
            tokenizer.encode('中立', add_special_tokens=False)[0]: 1,
            tokenizer.encode('相似', add_special_tokens=False)[0]: 2,
        }


class OCNLIMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('矛盾', add_special_tokens=False)[0]: 0,
            tokenizer.encode('中立', add_special_tokens=False)[0]: 1,
            tokenizer.encode('相似', add_special_tokens=False)[0]: 2,
        }


class AmazonMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('差', add_special_tokens=False)[0]: 0,
            tokenizer.encode('不好', add_special_tokens=False)[0]: 1,
            tokenizer.encode('一般', add_special_tokens=False)[0]: 2,
            tokenizer.encode('好', add_special_tokens=False)[0]: 3,
            tokenizer.encode('赞', add_special_tokens=False)[0]: 4,
        }


class BQMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('矛盾', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('相似', add_special_tokens=False)[0]: 1,  # positive
        }


class CCPMMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('A', add_special_tokens=False)[0]: 0,
            tokenizer.encode('B', add_special_tokens=False)[0]: 1,
            tokenizer.encode('C', add_special_tokens=False)[0]: 2,
            tokenizer.encode('D', add_special_tokens=False)[0]: 3,
        }


class TNewsMetric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode(label, add_special_tokens=False)[0]: i for i, label in enumerate(["房产", "汽车", "金融", "体育", "文化", "娱乐", "教育", "科技", "军事", "旅游", "世界", "农业", "股票", "游戏", "故事"])
        }


class C3Metric(BasicMetric):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__(pred, target, seq_len)
        self.label_map = {
            tokenizer.encode('A', add_special_tokens=False)[0]: 0,
            tokenizer.encode('B', add_special_tokens=False)[0]: 1,
            tokenizer.encode('C', add_special_tokens=False)[0]: 2,
            tokenizer.encode('D', add_special_tokens=False)[0]: 3,
        }