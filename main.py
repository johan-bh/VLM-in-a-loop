import os
import sys
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.dataset import RadiologyDataset
from utils.transforms import get_transforms

def check_create_paths(xml_dir: str = 'data/ecgen-radiology',
                 image_dir: str = 'data/radiology/extract',
                    save_dir: str = 'data/samples') -> None:
    
    """ Checks and creates the necessary directories for the dataset. """
    # Check if the XML directory exists
    if not os.path.exists(xml_dir):
        sys.exit(f"Error: XML directory '{xml_dir}' not found.")
        
    # Check if the image directory exists
    if not os.path.exists(image_dir):
        sys.exit(f"Error: Image directory '{image_dir}' not found.")
        
    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("All directories are present and correct.")
    return None

def main(xml_dir: str = 'data/ecgen-radiology', 
         image_dir: str = 'data/radiology/extract', 
         save_dir: str = 'data/samples',
         augment: bool = False) -> None:
    """
    Load and display the first sample from the RadiologyDataset.
    
    Args:
        xml_dir (str): Directory path where XML files are stored.
        image_dir (str): Directory path where image files are stored.
        augment (bool): Whether to use data augmentation transforms.
    """    
    # Initialize the dataset
    dataset: Dataset = RadiologyDataset(xml_dir=xml_dir, image_dir=image_dir, transform=get_transforms(augment))
    
    # Load the first sample
    sample: Dict[str, Any] = dataset[0]
    
    # Sample image
    image: Image.Image = sample['image']
    image.save(os.path.join(save_dir, 'sample_image.jpg'))
    
    # Sample report
    report: str = sample['report']
    print(report)

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load and display the first sample from the RadiologyDataset.")
    parser.add_argument("--xml_dir", type=str, default='data/ecgen-radiology', help="Directory path where XML files are stored.")
    parser.add_argument("--image_dir", type=str, default='data/radiology/extract', help="Directory path where image files are stored.")
    parser.add_argument("--augment", action="store_true", help="Whether to use data augmentation transforms.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main function
    main(xml_dir=args.xml_dir, image_dir=args.image_dir, augment=args.augment)
    