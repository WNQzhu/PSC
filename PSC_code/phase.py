import math
import os
import torch
import argparse
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--checkpoint_path', type=str, default="output-debug-tmp-4096-d-2e-4-mean-init-s-lnE")
    parser.add_argument('--trainable_params', type=str, default="embed,norm,group_context")
    args = parser.parse_args()
    return args


def main(args):
    path = args.checkpoint_path

    #self.up_phase   = nn.Parameter(torch.empty(self.half_hd, self.dim))
    #self.down_phase = nn.Parameter(torch.empty(self.dim, self.half_hd)) 
    #model.layers.2.self_attn.phase.up_phase
    #model.layers.2.self_attn.phase.down_phase
    #nn.init.kaiming_uniform_(self.up_phase, a = math.sqrt(5))
    #nn.init.zeros_(self.down_phase)
    
    params_list = []
    weights_trainable = {}
    #phase
    for idx in range(32):
        name = "model.layers.{}.self_attn.q_phase.down_phase".format(idx)
        #name = "model.q_phase.down_phase"
        t = nn.Parameter(torch.empty(32, 128, 128, dtype=torch.bfloat16))
        #t = nn.Parameter(torch.empty(128, 128, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(t, a = math.sqrt(5))
        weights_trainable[name] = t

    for idx in range(32):
        name = "model.layers.{}.self_attn.k_phase.down_phase".format(idx)        
        #name = "model.k_phase.down_phase"
        t = nn.Parameter(torch.empty(32, 128, 128, dtype=torch.bfloat16))
        #t = nn.Parameter(torch.empty(128, 128, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(t, a = math.sqrt(5))
        weights_trainable[name] = t
        
    for idx in range(32):
        name = "model.layers.{}.self_attn.q_phase.up_phase".format(idx)                
        #name = "model.q_phase.up_phase"
        t = nn.Parameter(torch.empty(32, 128, 128, dtype=torch.bfloat16))
        #t = nn.Parameter(torch.empty(128, 128, dtype=torch.bfloat16))
        nn.init.zeros_(t)
        weights_trainable[name] = t
        
    for idx in range(32):
        name = "model.layers.{}.self_attn.k_phase.up_phase".format(idx)                        
        #name = "model.k_phase.up_phase"
        t = nn.Parameter(torch.empty(32, 128, 128, dtype=torch.bfloat16))
        #t = nn.Parameter(torch.empty(128, 128, dtype=torch.bfloat16))
        nn.init.zeros_(t)
        weights_trainable[name] = t
    #q
    for idx in range(32):
        name = "model.layers.{}.self_attn.q_lora.A_lora".format(idx)
        t = nn.Parameter(torch.empty(8, 4096, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(t, a = math.sqrt(5))
        weights_trainable[name] = t
    for idx in range(32):
        name = "model.layers.{}.self_attn.q_lora.B_lora".format(idx)
        t = nn.Parameter(torch.empty(4096, 8, dtype=torch.bfloat16))
        nn.init.zeros_(t)
        weights_trainable[name] = t
    #k
    for idx in range(32):
        name = "model.layers.{}.self_attn.k_lora.A_lora".format(idx)
        t = nn.Parameter(torch.empty(8, 4096, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(t, a = math.sqrt(5))
        weights_trainable[name] = t
    for idx in range(32):
        name = "model.layers.{}.self_attn.k_lora.B_lora".format(idx)
        t = nn.Parameter(torch.empty(4096, 8, dtype=torch.bfloat16))
        nn.init.zeros_(t)
        weights_trainable[name] = t

    #v
    for idx in range(32):
        name = "model.layers.{}.self_attn.v_lora.A_lora".format(idx)
        t = nn.Parameter(torch.empty(8, 4096, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(t, a = math.sqrt(5))
        weights_trainable[name] = t
    for idx in range(32):
        name = "model.layers.{}.self_attn.v_lora.B_lora".format(idx)
        t = nn.Parameter(torch.empty(4096, 8, dtype=torch.bfloat16))
        nn.init.zeros_(t)
        weights_trainable[name] = t
        
    #o
    for idx in range(32):
        name = "model.layers.{}.self_attn.o_lora.A_lora".format(idx)
        t = nn.Parameter(torch.empty(8, 4096, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(t, a = math.sqrt(5))
        weights_trainable[name] = t
    for idx in range(32):
        name = "model.layers.{}.self_attn.o_lora.B_lora".format(idx)
        t = nn.Parameter(torch.empty(4096, 8, dtype=torch.bfloat16))
        nn.init.zeros_(t)
        weights_trainable[name] = t                
       
    for n, p in weights_trainable.items():
        print("n, p:", n, p)
    trainable_params = os.path.join(path, "trainable_params.safetensors")
    safe_save_file(weights_trainable, trainable_params, metadata={"format":"pt"})
    #torch.save(weights_trainable, trainable_params)

if __name__ == "__main__":
    args = parse_config()
    main(args)
