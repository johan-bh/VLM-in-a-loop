import os
import sys
from typing import Any, Dict, List

import torch
from loguru import logger
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset
from PIL import Image

from utils.dataset import RadiologyDataset
from utils.transforms import get_transforms

def configure_logger(log_file: str = 'logs/main.log') -> None:
    """ Configures the logger to write to a log file. """
    logger.remove()
    logger.add(log_file)
    return None

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
    
    logger.debug("All directories are present and correct.")
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
    # Configure the logger
    configure_logger()
    # Check and create the necessary directories
    check_create_paths(xml_dir, image_dir, save_dir)
    # Initialize the dataset
    logger.debug(f"Initializing {'and augmenting' if augment else ''} the dataset...")
    dataset: Dataset = RadiologyDataset(xml_dir=xml_dir,
                                        image_dir=image_dir, 
                                        transform=get_transforms(augment)
                                       )
    
    # Load the first sample
    sample: Dict[str, Any] = dataset[0]
    logger.debug(f"Dataset sample metadata: \n{sample['metadata']}")
    
    # Sample image
    image: Image.Image = sample['images']
    for i, image in enumerate(sample['images']):
        image = to_pil_image(image)
        image.save(os.path.join(save_dir, f'sample_image_{i+1}.jpg'))
    
    # Sample report
    report: Dict = sample['report']
    logger.debug("Sample report:") 
    for label, text in report.items():
        logger.debug(f"{label}: {text}")
    return None

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load and display the first sample from the RadiologyDataset.")
    parser.add_argument("--xml_dir", type=str, default='data/ecgen-radiology', help="Directory path where XML files are stored.")
    parser.add_argument("--image_dir", type=str, default='data/radiology', help="Directory path where image files are stored.")
    parser.add_argument("--augment", action="store_true", help="Whether to use data augmentation transforms.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main function
    main(xml_dir=args.xml_dir, image_dir=args.image_dir, augment=args.augment)
    