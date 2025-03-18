import torchvision.transforms as T

def get_transforms(augment: bool = False) -> T.Compose:
    """
    Create a set of data augmentation transforms.
    
    Args:
        augment (bool): Whether to use data augmentation transforms.
    
    Returns:
        T.Compose: A composition of data augmentation transforms.
    """
    if augment:
        # Data augmentation transforms
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Standard normalization transforms
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])