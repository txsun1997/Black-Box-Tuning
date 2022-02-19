import os
import numpy as np
import random
import sys
import onnx
import torch
import transformers
import argparse
from modeling_roberta import RobertaModel
from transformers import RobertaConfig, RobertaTokenizer
import onnxruntime as ort
from onnxruntime.transformers import optimizer
from typing import Callable, Dict, List, OrderedDict, Tuple


def export_onnx_model():
    if os.path.exists(exported_model_path):
        print(f'Found exported onnx model at {exported_model_path}.')
        return

    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(input_ids, attention_mask, prompt_embedding),
            f=exported_model_path,
            verbose=False,
            input_names=['input_ids', 'attention_mask', 'prompt_embedding'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'max_seq_len'},
                'attention_mask': {0: 'batch_size', 1: 'max_seq_len'},
                'prompt_embedding': {0: 'batch_size'},
                'logits': {0: 'batch_size', 1: 'max_seq_len'}
            },
            do_constant_folding=True,
            use_external_data_format=False,
            enable_onnx_checker=True,
            opset_version=12,
        )
        onnx_model = onnx.load(exported_model_path)
        onnx.checker.check_model(onnx_model)
        print('export finished')


def export_and_optimize_onnxruntime_model():
    export_onnx_model()

    optimized_model = optimizer.optimize_model(
        exported_model_path,
        model_type='bert',
        num_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
        opt_level=99,
        use_gpu=True
    )
    optimized_model.convert_float_to_float16()
    print('fp16 optimization finished')
    optimized_model.save_model_to_file(optimized_model_path)
    print(f'optimized model saved to {optimized_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='roberta-large', type=str, help='Pretrained model our BBT bases on.')
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='''Model batch size during export. Independent to batch size during inference. 
             Since batch size axis is dynamic, we recommend you use default value.'''
    )
    parser.add_argument(
        '--max_seq_len',
        default=128,
        type=int,
        help='''Model max sequence length during export. Independent to max sequence length during inference. 
             Since max sequence length axis is dynamic, we recommend you use default value.'''
    )
    parser.add_argument('--n_prompt_tokens', default=50, type=int, help='Number of prompt tokens during inference.')
    parser.add_argument('--prompt_embed_dim', default=1024, type=int, help='Prompt embedding dimension.')
    parser.add_argument(
        '--exported_model_name',
        default='model',
        type=str,
        help='File name of exported onnx model. No prefix'
    )
    parser.add_argument(
        '--optimized_model_name',
        default='optimized_model',
        type=str,
        help='File name of optimized onnx model. No prefix.'
    )
    args = parser.parse_args()

    if not os.path.exists('onnx_models'):
        os.mkdir('onnx_models')

    model_name = args.model_name
    bsz = args.batch_size
    max_seq_len = args.max_seq_len
    n_prompt_tokens = args.n_prompt_tokens
    prompt_embed_dim = args.prompt_embed_dim

    config = RobertaConfig.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).eval().cuda()
    input_ids = torch.randint(low=1, high=10000, size=(bsz, max_seq_len)).type(torch.int64).cuda()
    attention_mask = torch.ones(bsz, max_seq_len).type(torch.int64).cuda()
    prompt_embedding = torch.randn(size=(bsz, n_prompt_tokens, prompt_embed_dim)).cuda()
    exported_model_path = os.path.join('onnx_models', args.exported_model_name + '.onnx')
    optimized_model_path = os.path.join('onnx_models', args.optimized_model_name + '.onnx')
    export_and_optimize_onnxruntime_model()
