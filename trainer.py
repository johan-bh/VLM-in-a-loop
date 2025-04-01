import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from typing import Any, Dict
from loguru import logger

class VLMTrainerHF:
    """
    A Hugging Face Trainer wrapper for fine-tuning the Qwen2 VL 7B model using PEFT/LoRA.
    """
    def __init__(self, model: torch.nn.Module, tokenizer: Any, train_dataset: Any, eval_dataset: Any) -> None:
        """
        Initializes the trainer.
        
        Args:
            model (torch.nn.Module): The Qwen2 VL 7B model (with PEFT modifications).
            tokenizer (Any): Tokenizer for the model.
            train_dataset (Any): Training dataset.
            eval_dataset (Any): Evaluation dataset.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="epoch",
            eval_steps=1000,
            logging_steps=1000,
            save_steps=1500,
            num_train_epochs=10,
            weight_decay=0.01,
            learning_rate=2e-4,
            fp16=True,
            report_to="none"
        )
        self.data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
    
    def train(self) -> None:
        """
        Starts the training process.
        """
        logger.info("Starting training with HF Trainer...")
        train_result = self.trainer.train()
        self.trainer.save_model()
        logger.info("Training completed.")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the model.
        
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        logger.info("Starting evaluation...")
        metrics = self.trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
