import datasets
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
from transformers import T5Tokenizer


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=8)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings


class SST2Loader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "negative",
            1: "positive",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = 'text: %s %s . It was <extra_id_0> </s>' % (prompt, example['sentence'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]
        else:
            example['input_text'] = 'text: %s . It was <extra_id_0> </s>' % example['sentence']
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('glue', 'sst2', split=split)
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
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class YelpPLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "negative",
            1: "positive",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = 'text: %s %s . It was <extra_id_0> </s>' % (prompt, example['text'].replace("\\n", " "))
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]
        else:
            example['input_text'] = 'text: %s . It was <extra_id_0> </s>' % (example['text'].replace("\\n", " "))
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('yelp_polarity', 'plain_text', split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
            ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class AGNewsLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Tech"
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s <extra_id_0> News: %s </s>' % (prompt, example['text'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]
        else:
            example['input_text'] = '<extra_id_0> News: %s </s>' % example['text']
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]

        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('ag_news', 'default', split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class DBPediaLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Company",
            1: "Education",
            2: "Artist",
            3: "Athlete",
            4: "Office",
            5: "Transportation",
            6: "Building",
            7: "Natural",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "Written",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s [ Category: <extra_id_0> ] %s' % (prompt, example['content'].strip())
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]
        else:
            example['input_text'] = '[ Category: <extra_id_0> ] %s' % (example['content'].strip())
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]

        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('dbpedia_14', split=split)
        # dataset = datasets.load_dataset('./data/dbpedia.py', split=split)  # if you cannot reach the source of dbpedia, try this
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
            ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MRPCLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? <extra_id_0> , %s' % (prompt, example['sentence1'], example['sentence2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]

        else:
            example['input_text'] = '%s ? <extra_id_0> , %s' % (example['sentence1'], example['sentence2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]

        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('glue', 'mrpc', split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
            ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class RTELoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "No",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? <extra_id_0> , %s' % (prompt, example['sentence1'], example['sentence2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]

        else:
            example['input_text'] = '%s ? <extra_id_0> , %s' % (example['sentence1'], example['sentence2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]

        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('glue', 'rte', split=split)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
            ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle



class SNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "Maybe",
            2: "No",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? <extra_id_0> , %s' % (prompt, example['premise'], example['hypothesis'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? <extra_id_0> , %s' % (example['premise'], example['hypothesis'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['label']]

        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = datasets.load_dataset('snli', split=split)
        dataset = dataset.filter(lambda example: example['label'] in [0, 1, 2])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
            ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle