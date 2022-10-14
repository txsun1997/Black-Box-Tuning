import os
import datasets
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
import torch
from tqdm import tqdm

################################################
#   Don't Forget To Modify BasicLoader.path    #
################################################

def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
    mask_pos = []
    for input_ids in input_encodings['input_ids']:
        mask_pos.append(input_ids.index(tokenizer.mask_token_id))
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'mask_pos': mask_pos,
        'labels': target_encodings['input_ids'],
    }

    return encodings

class BasicLoader(Loader):

    meta_path = '/home/ma-user/work/zfhe/chineseeval'

    def __init__(self, tokenizer, n_prompt_tokens):
        self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens

    def convert_examples(self, example):
        raise NotImplementedError

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset(self.path, split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class ChnSentLoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'chnsenticorp/chnsenticorp.py')
        self.label2text = {
            0: "差",
            1: "好",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s 。 %s 。 总之很%s .' % (prompt, example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s 。 总之很%s 。' % (example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example


class THUCNewsLoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'THUCNews/thuc_news.py')
        self.label2text = {
            0: '体育',
            1: '娱乐',
            2: '房产',
            3: '教育',
            4: '时尚',
            5: '政治',
            6: '游戏',
            7: '社会',
            8: '科技',
            9: '经济'
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s 。 %s 。 这句话的主题是%s .' % (prompt, example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s 。 这句话的主题是%s 。' % (example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example


class LCQMCLoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'LCQMC/LCQMC.py')
        self.label2text = {
            0: '矛盾',
            1: '相似'
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = f'{prompt}"{example["text1"]}"与"{example["text2"]}"两句话的意思是{self.tokenizer.mask_token}的。'
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = f'"{example["text1"]}"与"{example["text2"]}"两句话的意思是{self.tokenizer.mask_token}的。'
            example['target_text'] = self.label2text[example['label']]
        return example


class CMNLILoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'CMNLI/cmnli.py')
        self.label2text = {
            0: '矛盾',
            1: '中立',
            2: '相似',
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = f'{prompt}"{example["text1"]}"与"{example["text2"]}"两句话的意思是{self.tokenizer.mask_token}的。'
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = f'"{example["text1"]}"与"{example["text2"]}"两句话的意思是{self.tokenizer.mask_token}的。'
            example['target_text'] = self.label2text[example['label']]
        return example



class OCNLILoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'ocnli/ocnli.py')
        self.label2text = {
            0: '矛盾',
            1: '中立',
            2: '相似',
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = f'{prompt}"{example["text1"]}"与"{example["text2"]}"两句话的意思是{self.tokenizer.mask_token}的。'
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = f'"{example["text1"]}"与"{example["text2"]}"两句话的意思是{self.tokenizer.mask_token}的。'
            example['target_text'] = self.label2text[example['label']]
        return example


class AmazonLoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'amazon/amazon.py')
        self.label2text = {
            0: '差',
            1: '不好',
            2: '一般',
            3: '好',
            4: '赞',
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s 。 %s 。 总之很%s .' % (prompt, example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s 。 总之很%s 。' % (example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example

class BQLoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'bq_corpus/bq_corpus.py')
        self.label2text = {
            0: '矛盾',
            1: '相似'
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = f'{prompt}"{example["text1"]}"与"{example["text2"]}"两句话的意思是{self.tokenizer.mask_token}的。'
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = f'"{example["text1"]}"与"{example["text2"]}"两句话的意思是{self.tokenizer.mask_token}的。'
            example['target_text'] = self.label2text[example['label']]
        return example

class CCPMLoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'CCPM/ccpm.py')
        self.label2text = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def convert_examples(self, example):
        option_seq = ''
        for i, option in enumerate(example['options']):
            option_seq += f'{self.label2text[i]}、{option};'
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = f'{prompt}在选项{option_seq}中，与语段"{example["document"]}"意思最接近的选项是{self.tokenizer.mask_token}。'
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = f'在选项{option_seq}中，与语段"{example["document"]}"意思最接近的选项是{self.tokenizer.mask_token}。'
            example['target_text'] = self.label2text[example['label']]
        return example


class TNewsLoader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'tnews/tnews.py')
        self.label2text = {
            i: label for i, label in enumerate(["房产", "汽车", "金融", "体育", "文化", "娱乐", "教育", "科技", "军事", "旅游", "世界", "农业", "股票", "游戏", "故事"])
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s 。 %s 。 这句话的主题是%s .' % (prompt, example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s 。 这句话的主题是%s 。' % (example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example


class C3Loader(BasicLoader):
    def __init__(self, tokenizer, n_prompt_tokens=50):
        super().__init__(tokenizer, n_prompt_tokens)
        self.path = os.path.join(self.meta_path, 'CCPM/ccpm.py')
        self.label2text = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def convert_examples(self, example):
        option_seq = ''
        for i, option in enumerate(example['options']):
            option_seq += f'{self.label2text[i]}、{option};'
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = f'{prompt}语段"{example["document"]}"。问题"{example["question"]}"。选项"{option_seq}"答案是{self.tokenizer.mask_token}。'
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = f'语段"{example["document"]}"。问题"{example["question"]}"。选项"{option_seq}"答案是{self.tokenizer.mask_token}。'
            example['target_text'] = self.label2text[example['label']]
        return example