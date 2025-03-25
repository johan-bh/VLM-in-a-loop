import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from typing import Any

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model with low_cpu_mem_usage to avoid meta tensors.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          low_cpu_mem_usage=True,
                                                        #   attn_implementation='flash_attention_2', # flash-attn requires CUDA installation specification
                                                        #   device_map="auto", # Causes wierd meta tensor errors
                                                          torch_dtype=torch.bfloat16)
        # Set up LoRA configuration for causal language modeling.
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
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
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
