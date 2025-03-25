import argparse
import os
from typing import Any, Dict

import torch
from loguru import logger
from transformers import AutoTokenizer
from utils.finetune_dataset import FineTuneRadiologyDataset
from utils.transforms import get_transforms
from models.qwen import Qwen2VLV7BModel
from trainer import VLMTrainerHF

def configure_logger(log_file: str = 'logs/main.log') -> None:
    """
    Configures the logger to write debug output to a file.
    
    Args:
        log_file (str): Path to the log file.
    """
    logger.remove()
    logger.add(log_file, level="DEBUG")

def check_create_paths(xml_dir: str, image_dir: str, save_dir: str) -> None:
    """
    Checks that required directories exist and creates the output directory if needed.
    
    Args:
        xml_dir (str): Path to XML files.
        image_dir (str): Path to image files.
        save_dir (str): Path to save outputs.
    """
    if not os.path.exists(xml_dir):
        raise FileNotFoundError(f"XML directory '{xml_dir}' not found.")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' not found.")
    os.makedirs(save_dir, exist_ok=True)
    logger.debug("All directories are present and correct.")

def main(xml_dir: str, image_dir: str, save_dir: str, augment: bool, batch_size: int, epochs: int) -> None:
    """
    Main function for fine-tuning the Qwen2 VL 7B model on radiology reports.
    
    Args:
        xml_dir (str): Directory containing XML files.
        image_dir (str): Directory containing images.
        save_dir (str): Directory to save outputs.
        augment (bool): Whether to apply data augmentation.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
    """
    configure_logger()
    check_create_paths(xml_dir, image_dir, save_dir)
    
    # Load Qwen2 VL 7B model (with LoRA) and tokenizer
    # model_wrapper = Qwen2VLV7BModel(model_name="Qwen/Qwen2.5-7B-Instruct")
    model_wrapper = Qwen2VLV7BModel(model_name="unsloth/Qwen2.5-7B-bnb-4bit")
    tokenizer = model_wrapper.tokenizer
    
    # Create fine-tuning dataset (using radiology reports)
    transform = get_transforms(augment)
    train_dataset = FineTuneRadiologyDataset(xml_dir=xml_dir, image_dir=image_dir, tokenizer=tokenizer, transform=transform, max_length=512)
    # For demonstration, using the same dataset for evaluation
    eval_dataset = train_dataset
    
    # Initialize the Hugging Face Trainer
    trainer = VLMTrainerHF(model=model_wrapper.model, tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset)
    
    # Run training and evaluation
    logger.info(f"Training for {epochs} epochs with batch size {batch_size} on device: {trainer.device}")
    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Final evaluation metrics: %s", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2 VL 7B with LoRA using PEFT on radiology reports.")
    parser.add_argument("--xml_dir", type=str, default="data/ecgen-radiology", help="Directory path for XML files.")
    parser.add_argument("--image_dir", type=str, default="data/radiology", help="Directory path for radiology images.")
    parser.add_argument("--save_dir", type=str, default="data/samples", help="Directory to save outputs.")
    parser.add_argument("--augment", action="store_true", help="Whether to apply image augmentation transforms.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()
    
    main(xml_dir=args.xml_dir, image_dir=args.image_dir, save_dir=args.save_dir, 
         augment=args.augment, batch_size=args.batch_size, epochs=args.epochs)