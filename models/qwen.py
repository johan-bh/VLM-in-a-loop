import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from typing import Any, List
import torch.nn as nn
import re

def get_target_layers(model: nn.Module) -> List[str]:
    """
    Iterates over the model's modules and collects the names of layers that match
    the target patterns: "visual.blocks.*.attn.qkv", "visual.blocks.*.attn.proj", and 
    "visual.blocks.*.mlp.up_proj".
    
    Args:
        model (nn.Module): The model to inspect.
    
    Returns:
        List[str]: A list of module names that match the target layers.
    """
    target_layers = []
    # Define regex patterns for target modules.
    patterns = [
        r"visual\.blocks\.\d+\.attn\.qkv",
        r"visual\.blocks\.\d+\.attn\.proj",
        r"visual\.blocks\.\d+\.mlp\.up_proj"
    ]
    
    for name, module in model.named_modules():
        for pattern in patterns:
            if re.fullmatch(pattern, name):
                target_layers.append(name)
                break  # No need to check further patterns if one matched.
    return target_layers

class Qwen2VLV7BModel(torch.nn.Module):
    """
    A wrapper for the Qwen2-VL-7B model with LoRA adapters via PEFT.
    
    This implementation loads a pretrained Qwen2-VL-7B model from Hugging Face 
    without a device map, ensuring that all parameters are fully loaded on the CPU 
    (instead of remaining on the meta device). This allows the HF Trainer to later move 
    the model to the target device without encountering meta tensor errors.
    """
    def __init__(self, 
                 model_name: str = "Qwen2-VL-7B", 
                 lora_r: int = 8, 
                 lora_alpha: int = 32, 
                 lora_dropout: float = 0.1) -> None:
        """
        Initializes the model with LoRA configurations.
        
        Args:
            model_name (str): Hugging Face model identifier.
            lora_r (int): LoRA rank.
            lora_alpha (int): LoRA alpha scaling factor.
            lora_dropout (float): Dropout probability for LoRA layers.
        """
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="cpu",
            # trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.tokenizer = self.processor.tokenizer
        # Set up LoRA configuration for causal language modeling.
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=get_target_layers(self.model),
        )
        self.model = get_peft_model(self.model, lora_config)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            
        Returns:
            Any: Model outputs.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_length: int = 128) -> Any:
        """
        Generates text from the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            max_length (int): Maximum generated length.
            
        Returns:
            Any: Decoded text output.
        """
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

class Qwen25VL3BModel(torch.nn.Module):
    """A wrapper for the Qwen/Qwen2.5-VL-3B-Instruct-AWQ model with LoRA adapters via PEFT."""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> None:
        """Initializes the model with LoRA configurations."""
        super().__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        # Set up LoRA configuration for causal language modeling.
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=get_target_layers(self.model),
        )
        self.model = get_peft_model(self.model, lora_config)
        # Ensure adapter parameters require grad
        for name, param in self.model.named_parameters():
            if any(target in name for target in lora_config.target_modules):
                param.requires_grad = True
        self.model.train()  # re-set training mode after modifications
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        """
        Generates text from the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            max_length (int): Maximum generated length.
            
        Returns:
            Any: Decoded text output.
        """
        # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # return outputs
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
