
import torch

import math
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from itertools import chain
import copy

from datasets import load_dataset
from tqdm import tqdm
from utils import parse_args
import smdistributed.modelparallel
import smdistributed.modelparallel.torch as smp

from utils import is_main_process,main_process_first,wait_for_everyone

@smp.step
def train_step(model, batch):
    loss = model(**batch)["loss"]
    model.backward(loss)
    return loss

@smp.step
def test_step(model, batch):
    output = model(**batch)
    return output


def main():
    args = parse_args()
    smp.init()

    text_column = "question"
    label_column = "answer"


    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            print(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            print(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    dataset = load_dataset(
        'csv', data_files={
        "train": args.train_file,
        "validation": args.validation_file,
        })
    

    def preprocess_function(examples):
        inputs = [prompt + tokenizer.eos_token for prompt in examples["text"]]

        model_inputs = tokenizer(inputs)
        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
        return model_inputs
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    with main_process_first(smp.rank()):
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        processed_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                desc=f"Grouping texts in chunks of {block_size}",
            )
     
    wait_for_everyone()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_sampler = torch.utils.data.DistributedSampler(
                train_dataset,
                shuffle=True,
                seed=args.seed,
                rank=smp.dp_rank(),
                num_replicas=smp.dp_size(),
                drop_last=True,
            )
    
    eval_sampler = torch.utils.data.DistributedSampler(
                eval_dataset,
                shuffle=True,
                seed=args.seed,
                rank=smp.dp_rank(),
                num_replicas=smp.dp_size(),
                drop_last=True,
            )

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,sampler=eval_sampler, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, pin_memory=True
    )

    print(next(iter(train_dataloader)))

    # creating model
    with smp.model_creation(
        tensor_parallelism=True,
        dtype=torch.bfloat16,
        flash_attention=True
        ):
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,cache_dir="/tmp",torch_dtype=torch.bfloat16)

    model = smp.DistributedModel(model, trace_device="gpu")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )


    transformer_layers = model.get_module().transformer.seq_layers

    smp.set_activation_checkpointing(
    transformer_layers, pack_args_as_tuple=True, strategy='each')

    wait_for_everyone()

    optimizer = smp.DistributedOptimizer(optimizer)

    device = torch.device(f"cuda:{smp.local_rank()}")


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    for epoch in range(args.num_train_epochs):

        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_step(model,batch)
            total_loss += loss.reduce_mean().detach().float()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)

        if is_main_process(smp.rank()):
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        model.eval()
        eval_preds = []
        eval_loss = 0
        for estep, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = test_step(model,batch)
            loss = outputs["loss"].reduce_mean()
            eval_loss += loss.detach().float()
            logits_mb = outputs["logits"]
            logits = torch.cat(tuple(logits_mb.outputs), dim=0)
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                 )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        if is_main_process(smp.rank()):
            print(f"{epoch=}: {eval_ppl=} {eval_epoch_loss=}")

    # save the checkpoint

     
    smp.save_checkpoint(args.checkpoint_dir,
                tag=f"gptneo_3b_model.pt",
                partial=False,
                model=model,
                optimizer=optimizer)
    print("saving the final model")

    wait_for_everyone()

    if is_main_process(smp.rank()):
        tokenizer.save_pretrained(args.checkpoint_dir)
    

    wait_for_everyone()

if __name__ == "__main__":
    main()
