import os

import numpy as np
import random
import sys
import onnx
import torch
import transformers
from modeling_roberta import RobertaModel
from transformers import RobertaConfig, RobertaTokenizer
import onnxruntime as ort
from onnxruntime.transformers import optimizer

if not os.path.exists('onnx_models'):
    os.mkdir('onnx_models')

model_name = 'roberta-large'
bsz = 32
max_seq_len = 128
n_prompt_tokens = 50
prompt_embed_dim = 1024
hidden_size = 1024
exported_model_path = './onnx_models/RobertaModel_large_add.onnx'
optimized_model_path = './onnx_models/RobertaModel_large_add_optimized.onnx'

config = RobertaConfig.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name).eval().cuda()
input_ids = torch.randint(low=1, high=10000, size=(bsz, max_seq_len)).type(torch.int64).cuda()
attention_mask = torch.ones(bsz, max_seq_len).type(torch.int64).cuda()
prompt_embedding = torch.randn(size=(bsz, n_prompt_tokens, prompt_embed_dim)).cuda()

# ort_option = ort.SessionOptions()
# ort_session = ort.InferenceSession(
#     optimized_model_path,
#     ort_option,
#     providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
# )

# model precision check
# logits = torch.empty((bsz, max_seq_len, hidden_size), dtype=torch.float32, device='cuda')
# 
# io_binding = ort_session.io_binding()
# io_binding.bind_input(name='input_ids', device_type='cuda', device_id=0, element_type=np.longlong,
#                       shape=input_ids.shape, buffer_ptr=input_ids.data_ptr())
# io_binding.bind_input(name='attention_mask', device_type='cuda', device_id=0, element_type=np.longlong,
#                       shape=attention_mask.shape, buffer_ptr=attention_mask.data_ptr())
# io_binding.bind_input(name='prompt_embedding', device_type='cuda', device_id=0, element_type=np.float32,
#                       shape=prompt_embedding.shape, buffer_ptr=prompt_embedding.data_ptr())
# io_binding.bind_output('logits', device_type='cuda', device_id=0, element_type=np.float32,
#                        shape=(bsz, max_seq_len, hidden_size), buffer_ptr=logits.data_ptr())
# ort_session.run_with_iobinding(io_binding)
# 
# print(model(input_ids.cuda(), attention_mask.cuda(), prompt_embedding.cuda()) - logits)

# export
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
            'prompt_embedding': {0: 'batch_size', 1: 'n_prompt_tokens', 2: 'prompt_embed_dim'},
            'logits': {0: 'batch_size', 1: 'max_seq_len'}
        },
        do_constant_folding=True,
        use_external_data_format=False,
        enable_onnx_checker=True,
        opset_version=12,
    )
#
onnx_model = onnx.load(exported_model_path)
onnx.checker.check_model(onnx_model)
print('export finished')

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