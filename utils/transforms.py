from torchvision import transforms
from typing import Any

def get_transforms(augment: bool = False) -> Any:
    """
    Returns a torchvision transform pipeline for preprocessing images.
    
    Args:
        augment (bool): Whether to include data augmentation.
        
    Returns:
        Any: A torchvision.transforms.Compose object.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
