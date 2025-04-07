import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import xml.etree.ElementTree as ET
import os
import glob
from tqdm import tqdm
import re

# Paths
XML_DIR = r'C:\Users\jbhan\Desktop\VLM-in-a-loop\data\ecgen-radiology'
IMG_DIR = r'C:\Users\jbhan\Desktop\VLM-in-a-loop\data\radiology'

# Load pre-trained ViT
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={0: "Normal", 1: "Abnormal"},
    label2id={"Normal": 0, "Abnormal": 1}
)
# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Extract labels from XML and map to available images
def get_data_mapping():
    """
    Create a mapping between available images and their labels from XML files
    Returns a list of (image_path, label) tuples
    """
    print("Creating dataset mapping...")
    data_mapping = []
    image_pattern = re.compile(r'CXR(\d+)_')
    available_images = glob.glob(os.path.join(IMG_DIR, "CXR*.png"))
    
    # Create a dict of available image IDs
    image_id_to_path = {}
    for img_path in available_images:
        img_filename = os.path.basename(img_path)
        match = image_pattern.match(img_filename)
        if match:
            img_id = match.group(1)
            image_id_to_path[img_id] = img_path
    
    print(f"Found {len(image_id_to_path)} unique image IDs")
    
    # Process XML files to extract labels
    xml_files = glob.glob(os.path.join(XML_DIR, "*.xml"))
    for xml_path in tqdm(xml_files, desc="Processing XML files"):
        try:
            # Extract ID from XML filename
            xml_basename = os.path.splitext(os.path.basename(xml_path))[0]
            
            # Check if we have a matching image
            if xml_basename in image_id_to_path:
                img_path = image_id_to_path[xml_basename]
                
                # Extract label
                tree = ET.parse(xml_path)
                root = tree.getroot()
                impression = root.find('.//AbstractText[@Label="IMPRESSION"]')
                
                if impression is not None and impression.text:
                    impression_text = impression.text.lower()
                    label = 0 if "normal" in impression_text else 1
                    data_mapping.append((img_path, label))
            
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
    
    print(f"Created dataset with {len(data_mapping)} entries")
    return data_mapping

# Dataset class
class ChestXrayDataset(Dataset):
    def __init__(self, data_mapping, processor):
        self.data_mapping = data_mapping
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data_mapping)
    
    def __getitem__(self, idx):
        img_path, label = self.data_mapping[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            # Process image using ViT processor
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze()
            
            return pixel_values, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample in case of error
            dummy = torch.zeros(3, 224, 224)
            return dummy, label

def main():
    # Get data mapping
    data_mapping = get_data_mapping()
    
    # Check class distribution
    labels = [label for _, label in data_mapping]
    normal_count = labels.count(0)
    abnormal_count = labels.count(1)
    total = len(labels)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {total}")
    print(f"Normal: {normal_count} ({normal_count/total*100:.2f}%)")
    print(f"Abnormal: {abnormal_count} ({abnormal_count/total*100:.2f}%)")
    
    # Create dataset
    dataset = ChestXrayDataset(data_mapping, processor)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Quick validation check after each epoch
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Validation"):
                images = images.to(device)
                outputs = model(pixel_values=images)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Evaluation"):
            images = images.to(device)
            outputs = model(pixel_values=images)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nFinal Evaluation Results:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Abnormal"]))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    
    # Save the model
    model.save_pretrained("chest_xray_vit_model")
    print("Model saved to 'chest_xray_vit_model'")

if __name__ == "__main__":
    main() 