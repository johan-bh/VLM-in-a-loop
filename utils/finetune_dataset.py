from typing import Any, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from utils.dataset import RadiologyDataset

class FineTuneRadiologyDataset(Dataset):
    """
    A dataset wrapper for fine-tuning the Qwen2 VL 7B model on radiology reports.
    
    Uses the RadiologyDataset to load reports and tokenizes the report text using the provided tokenizer.
    """
    def __init__(self, 
                 xml_dir: str, 
                 image_dir: str, 
                 tokenizer: PreTrainedTokenizer, 
                 transform: Any = None,
                 max_length: int = 512) -> None:
        """
        Initializes the fine-tuning dataset.
        
        Args:
            xml_dir (str): Directory of XML files.
            image_dir (str): Directory of image files.
            tokenizer (PreTrainedTokenizer): Tokenizer for report text.
            transform (Any): Image transformation pipeline.
            max_length (int): Maximum token length.
        """
        self.radiology_dataset = RadiologyDataset(xml_dir=xml_dir, image_dir=image_dir, transform=transform)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.radiology_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.radiology_dataset[idx]
        report_text = sample["report"]
        # Ensure report_text is a string; if not, convert it.
        if not isinstance(report_text, str):
            report_text = str(report_text)
        tokenized = self.tokenizer(
            report_text, 
            truncation=True, 
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Remove the batch dimension
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        # For a text-to-text fine-tuning setup, we use the report as both input and target.
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"]
        }
