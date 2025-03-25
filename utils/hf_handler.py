from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Any

def load_model_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """
    Loads a model and its tokenizer from Hugging Face.
    
    Args:
        model_name (str): The model identifier (name or path).
        
    Returns:
        Tuple[Any, Any]: The loaded model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
