import argparse
import torch
import random
import numpy as np
import os
import scipy.special as sp
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import set_seed, default_data_collator
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from typing import Dict, Optional, Sequence
import math
from dataclasses import dataclass
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import KLDivLoss, MSELoss
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
    apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))

        input_ids = torch.nn.utils.rnn.pad_sequence(torch.tensor(input_ids), batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(torch.tensor(labels), batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
        )
def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb:
        import wandb

        wandb.login()
    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
        # fsdp_plugin=fsdp_plugin,
    )
    accelerator.init_trackers(project_name=args.wandb, init_kwargs={"wandb":{"name":args.output_dir.split("/")[-1]}})
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    try:
        train_dataset = load_dataset(args.dataset)
        auxiliary_dataset = load_dataset(args.dataset2)
        s2l_dataset = load_dataset(args.dataset3)
    except:
        train_dataset = load_from_disk(args.dataset)
        auxiliary_dataset = load_from_disk(args.dataset2)
        s2l_dataset = load_from_disk(args.dataset3)
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map=accelerator.device,
        torch_dtype=torch.bfloat16,
        rope_theta=args.rope_theta,
        _attn_implementation="flash_attention_2",
        max_position_embeddings = args.seq_length,
        use_cache=False
    )

    model_type = (
        "llama"
    )
    apply_seq_parallel_monkey_patch(args.parallel_mode, model_type)

    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    to_remove = [col for col in train_dataset.column_names if col != "input_ids"]
    train_dataset = train_dataset.remove_columns(to_remove)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    auxiliary_dataset = auxiliary_dataset.remove_columns(to_remove)
    auxiliary_dataset = auxiliary_dataset.shuffle(seed=args.seed)
    s2l_dataset = s2l_dataset.remove_columns(to_remove)
    s2l_dataset = s2l_dataset.shuffle(seed=args.seed)
    print("Dataset Size:", len(train_dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        if tokenizer.unk_token is None:
            tokenizer.pad_token = tokenizer.bos_token # e.x. falcon
        else:
            tokenizer.pad_token = tokenizer.unk_token # e.x. llama
    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size,
    )

    auxiliary_loader = DataLoader(
        auxiliary_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size2,
    )
    s2l_loader = DataLoader(
        s2l_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size3,
    )
    if args.learning_rate != 2e-5:
        accelerator.print(f"Warning: You also need to modify accelerate_configs/zero3_offload.json to change the learning rate")
    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(
        optim,
        num_training_steps=args.max_train_steps,
        total_num_steps=args.max_train_steps,
    )
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator)
    model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    iter3 = iter(s2l_loader)
    iter2 = iter(auxiliary_loader)
    iter1 = iter(train_loader)
    model.train()
    loss_func = CrossEntropyLoss(inplace_backward=True)
    flag=False
    # for step, batch in enumerate(train_loader):
    
    select_step = [0,2,4,6,8,10,12,14]
    
    pose_step = [1,3]
    for step in range(20000):
        if step%args.gradient_accumulate_every in select_step:
            batch = next(iter1)
            input_ids = batch["input_ids"][..., : args.seq_length + 1][..., :-1]
            target_ids = batch["input_ids"][..., : args.seq_length + 1][..., 1:]

            position_ids = (
                torch.arange(args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
            )
            # shard the input_ids according to the world size and rank according to zig zag attention

            prepared = prepare_seq_parallel_inputs(
                args.parallel_mode,
                input_ids,
                position_ids,
                target_ids,
                accelerator.process_index,
                accelerator.num_processes,
                accelerator.device,
            )


            local_input_ids = prepared["local_input_ids"]
            local_target_ids = prepared['local_target_ids']
            local_position_ids = prepared["local_position_ids"]
        elif step%args.gradient_accumulate_every in pose_step:
        # else:
            batch3 = next(iter3)
            input_ids4 = batch3["input_ids"][..., : 8192]
            target_ids4 = batch3["input_ids"][..., : 8193][..., 1:]

            random.seed(step)
            np.random.seed(step)
            # position_ids4 =CREAM(8192, 32768, input_ids4)
            # position_ids4 = torch.tensor(position_ids4)
            # position_ids4= torch.stack(position_ids4,dim=0)
            position_ids5 = []
            for id_idx in range(batch3["input_ids"].shape[0]):
                position_id = torch.arange(8192)
                position_ids5.append(position_id)
            position_ids5= torch.stack(position_ids5,dim=0)
            # position_ids2 = position_ids3
            # shard the input_ids according to the world size and rank according to zig zag attention
            
            prepared4 = prepare_seq_parallel_inputs(
                'zigzag_ring_attn2',
                input_ids4,
                position_ids5,
                target_ids4,
                accelerator.process_index,
                accelerator.num_processes,
                accelerator.device,
                position_ids5
            )
            local_input_ids4 = prepared4["local_input_ids"]
            local_target_ids4 = prepared4['local_target_ids']
            local_position_ids4 = prepared4["local_position_ids"]
            local_position_ids5 = prepared4["local_position_ids2"]
            loss_log = None
        else:
            batch2 = next(iter2)
            input_ids2 = batch2["input_ids"][..., : 1024]
            target_ids2 = batch2["input_ids"][..., : 1025][..., 1:]
            position_ids2 = []
            random.seed(step)
            for id_idx in range(batch2["input_ids"].shape[0]):
                position_id = torch.arange(1024)
                position_ids2.append(position_id)
            position_ids2= torch.stack(position_ids2,dim=0)

            position_ids3 = []
            for id_idx in range(batch2["input_ids"].shape[0]):
                position_id = torch.arange(1024)
                position_ids3.append(position_id)
            position_ids3= torch.stack(position_ids3,dim=0)

            prepared2 = prepare_seq_parallel_inputs(
                'zigzag_ring_attn2',
                input_ids2,
                position_ids2,
                target_ids2,
                accelerator.process_index,
                accelerator.num_processes,
                accelerator.device,
                position_ids3
            )

      
            local_input_ids2 = prepared2["local_input_ids"]
            local_target_ids2 = prepared2['local_target_ids']
            local_position_ids2 = prepared2["local_position_ids"]
            local_position_ids3 = prepared2["local_position_ids2"]
        #     loss_log = None
        with accelerator.accumulate(model):
            if step%args.gradient_accumulate_every in select_step:
                output = model(
                    local_input_ids,
                    position_ids=local_position_ids,
                    output_hidden_states=False
                )
                logits = output.logits
                loss = loss_func(
                    logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)
                )
                accelerator.backward(loss)
                del logits 
                del output 
            elif step%args.gradient_accumulate_every in pose_step:
                output2 = model(
                    local_input_ids4,
                    position_ids=local_position_ids4,
                    output_hidden_states=True
                )
                logits2 = output2.logits

                loss3 = loss_func(
                    logits2.reshape(-1, logits2.shape[-1]), local_target_ids4.reshape(-1)
                )
                
                accelerator.backward(loss3)
                del logits2
                del output2
            else:
                output2 = model(
                    local_input_ids2,
                    position_ids=local_position_ids2,
                    output_hidden_states=True
                )
                logits2 = output2.logits

                loss2 = loss_func(
                    logits2.reshape(-1, logits2.shape[-1]), local_target_ids2.reshape(-1)
                )
                
                accelerator.backward(loss2)
                del logits2
                del output2
            

            optim.step()
            scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            if loss_log is not None:
                progress_bar.set_postfix(loss_log)
            completed_steps += 1

        if completed_steps >= args.max_train_steps:
            break

    accelerator.print(f"Training Finished")
    accelerator.end_training()
    try:
        if args.output_dir is not None:
            accelerator.print(f"Saving model to {args.output_dir}")
            accelerator.wait_for_everyone()
            state_dict = accelerator.get_state_dict(model)
            accelerator.unwrap_model(model).save_pretrained(
                f"{args.output_dir}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,  
            )
        
            accelerator.print(f"Saving Finished")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--batch-size2", type=int, default=1)
    args.add_argument("--batch-size3", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--rope-theta", type=float, default=5000000)
    args.add_argument("--model", type=str)
    args.add_argument(
        "--dataset",
        type=str,
    )
    args.add_argument(
        "--dataset2",
        type=str,
    )
    args.add_argument(
        "--dataset3",
        type=str,
    )
    args.add_argument("--seq-length", type=int, default=32768)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["zigzag_ring_attn", "dist_flash_attn", "ulysses_attn", "data_parallel"],
    )
    main(args.parse_args())
