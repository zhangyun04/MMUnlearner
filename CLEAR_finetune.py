import sys
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel
import glob
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))
from datasets import load_dataset
import torch
import json
import argparse
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, MllamaForConditionalGeneration, AutoTokenizer,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.data_preprocess import LLAVA_multimodal_Dataset, train_collate_fn_mllmu, train_collate_mllmu_ansonly,Vanilla_LLaVA_Dataset
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from accelerate import Accelerator
import torch


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model',"visual"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Example usage:
def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    # Check if running in distributed mode - if so, don't use device_map
    # In distributed training, Accelerator will handle device placement
    is_distributed = os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") is not None
    device_map = None if is_distributed else "auto"
    
    if "llava" in args.model_id:
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    elif "llama" in args.model_id.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map=device_map,
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
        )
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    elif "qwen" in args.model_id.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, 
            device_map=device_map, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "left"  # Left padding required for Flash Attention 2
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor



######################### Accelerate Version #################################
def main(args):
    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)
    tokenizer = processor.tokenizer
    print("Tokenizer Length: ", len(tokenizer))

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
    
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Construct the dataset path - handle both relative and absolute paths
    dataset_path = args.data_split_dir
    if not os.path.isabs(dataset_path):
        # If relative path, make it relative to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, dataset_path)
    
    full_tofu_path = os.path.join(dataset_path, "full+tofu")
    
    if not os.path.exists(full_tofu_path):
        raise FileNotFoundError(f"Dataset directory not found: {full_tofu_path}")
    
    # Find all parquet files in the directory
    parquet_files = glob.glob(os.path.join(full_tofu_path, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {full_tofu_path}")
    
    # Load dataset from parquet files - data_files can be a list of file paths
    tofu_df = load_dataset("parquet", data_files=parquet_files, split="train")

    multimodal_tofu_dataset = CLEAR_Dataset(data=tofu_df,mode=NONE_MODE)
    multimodal_caption_dataset = CLEAR_Dataset(data=tofu_df,mode=CAPTION_MODE)
    multimodal_full_dataset=torch.utils.data.ConcatDataset([multimodal_tofu_dataset, multimodal_caption_dataset])

    if args.ans_only:
        train_collate_function = train_collate_clear_ansonly
    else:
        train_collate_function = train_collate_clear
    
    # Accelerator setup - will handle distributed training automatically
    # For PEFT models, we need to set find_unused_parameters=True because only LoRA parameters are trainable
    # This tells DDP to handle unused parameters (frozen base model parameters) correctly
    use_find_unused = isinstance(model, PeftModel)
    
    # For Accelerate 1.10.1, we need to manually wrap the model with DDP after prepare
    # or use a workaround. Let's use Accelerate's default and then manually fix it
    accelerator = Accelerator()
    
    # Print dataset info and training configuration
    if accelerator.is_main_process:
        print(f"Dataset sizes - TOFU: {len(multimodal_tofu_dataset)}, Caption: {len(multimodal_caption_dataset)}")
        print(f"Total dataset size: {len(multimodal_full_dataset)}")
        if args.ans_only:
            print("Answer only mode enabled.")
        else:
            print("Answer only mode disabled.")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Device: {accelerator.device}")
        print(f"Distributed training: {accelerator.distributed_type}")
        print(f"Effective batch size: {args.batch_size * accelerator.num_processes}")
        if args.gradient_accumulation:
            print("Gradient accumulation enabled.")
        else:
            print("Gradient accumulation disabled.")
    
    # Use accelerator.device for collate function
    device = accelerator.device
    if processor:
        train_dataloader = DataLoader(
            multimodal_full_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor, device, True)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")
    
    if args.gradient_accumulation:
        accumulation_steps = 4  # Adjust based on memory
        model.gradient_checkpointing_enable()

    # For PEFT models, parameters() already returns only trainable parameters
    # PEFT automatically sets requires_grad correctly
    # Ensure model is in training mode before creating optimizer
    model.train()
    
    # Get trainable parameters - for PEFT models, this should only include LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if accelerator.is_main_process and len(trainable_params) > 0:
        print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = AdamW(trainable_params, lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    # Prepare model, optimizer, dataloader, and scheduler for distributed training
    # For PEFT models, we need to manually wrap with DDP to set find_unused_parameters
    # Check if we're in distributed mode
    is_distributed = accelerator.distributed_type.name == "MULTI_GPU" if hasattr(accelerator.distributed_type, 'name') else str(accelerator.distributed_type) != "NO"
    
    if use_find_unused and is_distributed:
        # Manually prepare optimizer, dataloader, and scheduler (but not model)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler
        )
        # Manually wrap model with DDP, setting find_unused_parameters=True
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch.distributed as dist
        model = model.to(accelerator.device)
        # Get the process group from accelerator
        process_group = accelerator.state.process_group if hasattr(accelerator.state, 'process_group') else None
        model = DDP(
            model, 
            device_ids=[accelerator.local_process_index] if accelerator.device.type == "cuda" else None,
            output_device=accelerator.device if accelerator.device.type == "cuda" else None,
            find_unused_parameters=True,
            process_group=process_group
        )
        if accelerator.is_main_process:
            print("Manually wrapped PEFT model with DDP (find_unused_parameters=True)")
    else:
        # Normal prepare for non-PEFT models or single GPU
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        if args.gradient_accumulation:
            for step, batch in enumerate(progress_bar):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = (outputs.loss / accumulation_steps)
                    # loss = outputs.loss
                    accelerator.backward(loss)
                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  #
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))
            # lr_scheduler.step()
            avg_loss = total_loss / len(train_dataloader)
            if accelerator.is_main_process:
                print(f"Epoch {epoch + 1} Loss: {avg_loss}")
        else:
            for batch in progress_bar:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))

            avg_loss = total_loss / len(train_dataloader)
            if accelerator.is_main_process:
                print(f"Epoch {epoch + 1} Loss: {avg_loss}")

        # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.save_pretrained(args.save_dir)
        print(f"Model saved to: {args.save_dir}")

if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="data/CLEAR", help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--ans_only", type=bool, default=False, help="Answer only for training")

    args = parser.parse_args()

    # Call main function
    main(args)
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)