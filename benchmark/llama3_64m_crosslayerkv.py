import sys
import os

SOURCE_DIR = os.path.dirname(os.path.abspath(__name__))
sys.path.insert(0, SOURCE_DIR)

from more_memory_efficient.modeling_llama_crosslayerkv import LlamaForCausalLM as LlamaForCausalLM_crosslayerkv
from more_memory_efficient.cache import CrossLayerKVCache
from transformers import LlamaForCausalLM, AutoConfig
import torch
import time
import sys
import json

def main(max_length = 100):
    config = AutoConfig.from_pretrained('mesolitica/llama-3-8b-8192-hf')
    config.hidden_size = 512
    config.intermediate_size = 2048
    config.num_attention_heads = 16
    config.num_hidden_layers = 8
    config.vocab_size = 32000
    config.sharing_factor = 3

    model = LlamaForCausalLM_crosslayerkv(config).type(torch.bfloat16)
    _ = model.cuda()

    
    inputs = torch.tensor([[1]]).cuda()
    # warmup
    cache = CrossLayerKVCache()
    model.generate(input_ids = inputs, max_length = 2, past_key_values = cache)
    
    before = time.time()
    cache = CrossLayerKVCache()
    model.generate(input_ids = inputs, max_length = max_length, past_key_values = cache)
    after = time.time()
    with open(f'llama3_64_crosslayerkv-{max_length}.data', 'w') as fopen:
        json.dump({
            'time taken':after - before,
            'memory use': torch.cuda.max_memory_allocated(device=None),
        }, fopen)

if __name__ == "__main__":
    max_length = int(sys.argv[1])
    main(max_length=max_length)
