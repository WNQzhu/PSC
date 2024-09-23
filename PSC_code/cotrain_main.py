import os
import collections
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
import pdb

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from llama_attn_replace import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from lora_llama import LlamaForCausalLM
from safetensors.torch import load_file as safe_load_file

from datasets import load_dataset
from dataset import PretrainDataset
from collator import PretrainCollator

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    train_file: str = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    group_size: int = field(
        default=4096,
        metadata={"help": "The size of each group"},
    )
    context_len: int = field(
        default=1,
        metadata={"help": "the size of context len"},)
    last_k: int = field(
        default=80000000,
        metadata = {"help": "using last K group data"},
    )
    tau: float = field(
        default=1.0,
        metadata = {"help": "using tau to select items"})
    yarn_factor: float = field(
        default=None,
        metadata = {"help": "yarn factor"})
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    lora_dir: str = field(
        default="",
        metadata={"help": "Additional lora params"},)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    else:
        assert model_args.model_type == "llama", "Only support llama and gpt-neox for now"
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    yarn_factor = None
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            #config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
            #config.rope_scaling = {"type": "yarn", "factor": scaling_factor} 

           
            if training_args.yarn_factor is None:
                yarn_factor = scaling_factor
    use_pi = False
    if use_pi:
        config.rope_scaling = {"type": "linear", "factor": 16}
        config.max_position_embeddings = 65536
        #config.rope_scaling = {"type": "linear", "factor": 8.0}
        #config.max_position_embeddings = 32768
    use_yarn = True
    if use_yarn:
        config.rope_scaling = {"type": "yarn", "factor": 16.0, "original_max_position_embeddings": 4096}
        config.max_position_embeddings = 65536
        #config.rope_scaling = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}
        #config.max_position_embeddings = 32768
        
                
    setattr(config, "group_size", training_args.group_size)
    setattr(config, "last_k", training_args.last_k)
    setattr(config, "context_len", training_args.context_len)
    setattr(config, "tau", training_args.tau)
    #if training_args.yarn_factor is not None:
    #    yarn_factor = training_args.yarn_factor        
    setattr(config, "yarn_factor", yarn_factor)
    print("train config is: ", config)
    # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    # config._attn_implemention = "eager"
    # config.attn_implementaion = "eager"
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    #init_model_group_context(model, input_embeddings_avg)

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()
    train_dataset = PretrainDataset(training_args.train_file, tokenizer, training_args.model_max_length)
    data_collator = PretrainCollator(tokenizer, training_args.model_max_length, -100)
    #dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=training_args.cache_dir)
    #dataset = dataset.map(partial(tokenize_fn,tokenizer),batched=True, num_proc=128, remove_columns=["text", "meta"])

    if rank == 0:
        barrier()

    print(train_dataset)

    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if True:
        #+begin load lora
        lora_params = os.path.join(training_args.lora_dir, "trainable_params.safetensors")
        if os.path.isfile(lora_params):
            trainable_param_dict = safe_load_file(lora_params)
            new_param_dict = collections.defaultdict()
            print("trainble_params_dict: ", trainable_param_dict)
            
            for n, p in trainable_param_dict.items():
                #print("dtype: ", p.dtype, p.device, model.device)
                new_param_dict[n] = p.to(model.device)
                #print("dtype2: ", p.dtype, p.device)
            print("2trainable_params_dict: ", new_param_dict)
            model.load_state_dict(new_param_dict, strict=False)
        else:
            raise ValueError("lora params file not found")
    

    if False and training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
    for n, p in model.named_parameters():
        p.requires_grad_(False)       
    # enable trainable params    
    [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
    print("get_peft_model: ", model)
    print("print all np parameters:")
    for n, p in model.named_parameters():
        print(n, p)
    print("config: ", config)
    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator)
    #pdb.set_trace()
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    model.base_model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
