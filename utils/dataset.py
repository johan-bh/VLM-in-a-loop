import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from PIL import Image
from torch.utils.data import Dataset

class RadiologyDataset(Dataset):
    """
    A custom Dataset for loading radiology images and corresponding reports/metadata 
    from XML files in the OpenI Chest X-ray Collection.
    
    The dataset expects the following file structure:
        /root
         ├── ecgen-radiology/
         │    ├── 1.xml
         │    ├── 2.xml
         │    └── ... (up to 3999.xml)
         └── radiology/
              └── extract/
                   ├── CXR2_IM-0652-1001.jpg
                   ├── CXR2_IM-0652-2001.jpg
                   └── ...
    
    Attributes:
        xml_dir (str): Directory path where XML files are stored.
        image_dir (str): Directory path where image files are stored.
        transform (Optional[Any]): Optional transformation to apply to the images.
        xml_files (List[str]): List of XML file paths sorted alphabetically.
    """
    def __init__(self, xml_dir: str, image_dir: str, transform: Optional[Any] = None) -> None:
        """
        Initialize the RadiologyDataset.
        
        Args:
            xml_dir (str): Directory where the XML files are located.
            image_dir (str): Directory where the image files are located.
            transform (Optional[Any]): Optional transform (e.g., torchvision.transforms) to apply to the images.
        """
        self.xml_dir: str = xml_dir
        self.image_dir: str = image_dir
        self.transform: Optional[Any] = transform
        self.xml_files: List[str] = sorted([
            os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')
        ])

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of XML files (and thus samples) in the dataset.
        """
        return len(self.xml_files)
    
    def _parse_xml(self, xml_file: str) -> Dict[str, Any]:
        """
        Parse an XML file to extract the report text, metadata, and corresponding image file paths.
        
        Args:
            xml_file (str): Path to the XML file.
        
        Returns:
            Dict[str, Any]: A dictionary with keys:
                - "report": Combined text of all AbstractText elements.
                - "metadata": A dictionary of metadata (e.g., title, article_date, specialty).
                - "image_paths": A list of absolute paths for images referenced in the XML.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract report text from all AbstractText elements under Abstract
        report_texts: List[str] = []
        abstract = root.find('.//Abstract')
        if abstract is not None:
            for abstract_text in abstract.findall('AbstractText'):
                if abstract_text.text:
                    report_texts.append(abstract_text.text.strip())
        report: str = " ".join(report_texts)
        
        # Extract basic metadata such as article_date, title, and specialty
        metadata: Dict[str, Any] = {}
        article_date_elem = root.find('.//ArticleDate')
        if article_date_elem is not None:
            year = article_date_elem.find('Year').text if article_date_elem.find('Year') is not None else ""
            month = article_date_elem.find('Month').text if article_date_elem.find('Month') is not None else ""
            day = article_date_elem.find('Day').text if article_date_elem.find('Day') is not None else ""
            metadata['article_date'] = f"{year}-{month}-{day}".strip("-")
        
        title_elem = root.find('.//ArticleTitle')
        if title_elem is not None and title_elem.text:
            metadata['title'] = title_elem.text.strip()
        
        specialty_elem = root.find('.//specialty')
        if specialty_elem is not None and specialty_elem.text:
            metadata['specialty'] = specialty_elem.text.strip()
        
        # Extract image file paths from parentImage panels
        image_paths: List[str] = []
        for parent in root.findall('parentImage'):
            panel = parent.find('panel')
            if panel is not None:
                url_elem = panel.find('url')
                if url_elem is not None and url_elem.text:
                    # Get the basename of the file from the URL and join with the image directory
                    image_file = os.path.basename(url_elem.text.strip())
                    image_path = os.path.join(self.image_dir, image_file)
                    image_paths.append(image_path)
        
        return {"report": report, "metadata": metadata, "image_paths": image_paths}
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve the sample corresponding to the given index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - "images": List of loaded PIL.Image objects (or transformed images).
                - "report": The extracted report text.
                - "metadata": Dictionary of metadata information.
        """
        xml_file: str = self.xml_files[idx]
        data: Dict[str, Any] = self._parse_xml(xml_file)
        
        images: List[Any] = []
        for image_path in data["image_paths"]:
            # Open the image and convert to RGB
            img = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        
        return {
            "images": images,
            "report": data["report"],
            "metadata": data["metadata"],
        }

if __name__ == "__main__":
    # Test the RadiologyDataset class
    import torchvision.transforms as T
    from pprint import pprint

    # Initialize the dataset
    xml_dir = "/VLM-in-a-loop/data/ecgen-radiology"
    image_dir = "/VLM-in-a-loop/data/radiology/extract"
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    dataset = RadiologyDataset(xml_dir, image_dir, transform=transform)

    # Test the dataset
    sample = dataset[0]
    print("Report text:", sample["report"])
    print("Metadata:")
    pprint(sample["metadata"])
    print("Images:")
    for i, img in enumerate(sample["images"]):
        print(f"Image {i + 1}: {img.shape}")
        if i >= 5: # Only show the first 5 images
            break