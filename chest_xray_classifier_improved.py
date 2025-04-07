import torch
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor, AdamW
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from PIL import Image
import xml.etree.ElementTree as ET
import os
import glob
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import random

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# Paths
XML_DIR = r'C:\Users\jbhan\Desktop\VLM-in-a-loop\data\ecgen-radiology'
IMG_DIR = r'C:\Users\jbhan\Desktop\VLM-in-a-loop\data\radiology'

# Load pre-trained ViT - use a more appropriate model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",  # Using model pre-trained on ImageNet-1k instead of 21k
    num_labels=2,
    id2label={0: "Normal", 1: "Abnormal"},
    label2id={"Normal": 0, "Abnormal": 1},
    ignore_mismatched_sizes=True  # Add this to handle the classifier layer size mismatch
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
            # If we have multiple images for the same ID, keep only one
            if img_id not in image_id_to_path:
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

# Enhanced data augmentation
class ChestXrayDataset(Dataset):
    def __init__(self, data_mapping, processor, augment=False):
        self.data_mapping = data_mapping
        self.processor = processor
        self.augment = augment
        
        # Enhanced transformations for training
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize larger, then crop
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            # For validation/testing just resize
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
            
            # Apply transformations
            if self.augment:
                image_tensor = self.transform(image)
                # Convert to format expected by ViT processor
                # The processor expects a PIL image, not a tensor
                inputs = self.processor(images=image, return_tensors="pt")
            else:
                inputs = self.processor(images=image, return_tensors="pt")
            
            pixel_values = inputs['pixel_values'].squeeze()
            return pixel_values, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample in case of error
            dummy = torch.zeros(3, 224, 224)
            return dummy, label

def plot_training_history(train_losses, val_accuracies, save_path="training_history.png"):
    """Plot and save training history"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    
    # Stratified train-validation-test split
    np.random.shuffle(data_mapping)
    
    # Sort by label to stratify
    normal_samples = [d for d in data_mapping if d[1] == 0]
    abnormal_samples = [d for d in data_mapping if d[1] == 1]
    
    # Split each class 
    train_size_normal = int(len(normal_samples) * 0.7)
    val_size_normal = int(len(normal_samples) * 0.15)
    
    train_size_abnormal = int(len(abnormal_samples) * 0.7)
    val_size_abnormal = int(len(abnormal_samples) * 0.15)
    
    # Create stratified splits
    train_data = normal_samples[:train_size_normal] + abnormal_samples[:train_size_abnormal]
    val_data = normal_samples[train_size_normal:train_size_normal+val_size_normal] + \
               abnormal_samples[train_size_abnormal:train_size_abnormal+val_size_abnormal]
    test_data = normal_samples[train_size_normal+val_size_normal:] + \
                abnormal_samples[train_size_abnormal+val_size_abnormal:]
    
    # Shuffle the splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Create datasets with appropriate augmentation
    train_dataset = ChestXrayDataset(train_data, processor, augment=True)
    val_dataset = ChestXrayDataset(val_data, processor, augment=False)
    test_dataset = ChestXrayDataset(test_data, processor, augment=False)
    
    # Calculate class weights for weighted loss
    class_counts = [normal_count, abnormal_count]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    sample_weights = [class_weights[label] for _, label in train_data]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Training setup with weighted loss
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Use weighted loss for imbalanced dataset
    weights = torch.tensor([1.0, normal_count / abnormal_count]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    # Track metrics
    train_losses = []
    val_accuracies = []
    
    # Training loop
    epochs = 10  # Increase epochs
    best_val_accuracy = 0
    patience = 5  # Early stopping patience
    patience_counter = 0
    
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                outputs = model(pixel_values=images)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_accuracies.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load("best_model.pt"))
    
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
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nFinal Evaluation Results:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Abnormal"]))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    
    # Save the model
    model.save_pretrained("chest_xray_vit_model_improved")
    print("Model saved to 'chest_xray_vit_model_improved'")

if __name__ == "__main__":
    main() 