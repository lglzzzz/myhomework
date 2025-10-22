
hw3.ipynb
#%% 
import os
import xml.etree.ElementTree as ET
import numpy as np

# VOC2012 has 20 object categories
CLASS_MAP = {
    "person": 0, "bird": 1, "cat": 2, "cow": 3, "dog": 4, "horse": 5, "sheep": 6,
    "aeroplane": 7, "bicycle": 8, "boat": 9, "bus": 10, "car": 11, "motorbike": 12,
    "train": 13, "bottle": 14, "chair": 15, "diningtable": 16, "pottedplant": 17,
    "sofa": 18, "tvmonitor": 19
}

def parse_annotation(xml_file):
    """Parse one VOC XML file and return [class_id, x, y, w, h] as a NumPy array."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]

        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        w = xmax - xmin
        h = ymax - ymin
        boxes.append([cls_id, xmin, ymin, w, h])

    return np.array(boxes)

def extract_all_annotations(annotation_dir, output_dir):
    """Parse all XML files and save each result as a .npy file."""
    os.makedirs(output_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith(".xml")]

    for xml_file in xml_files:
        xml_path = os.path.join(annotation_dir, xml_file)
        boxes = parse_annotation(xml_path)
        img_id = os.path.splitext(xml_file)[0]
        np.save(os.path.join(output_dir, f"{img_id}.npy"), boxes)

    print(f"✅ Processed {len(xml_files)} XML files. Results saved in '{output_dir}/'")

def verify_output(output_dir, num_samples=3):
    """ 
    Randomly verify a few .npy files to ensure they have the correct shape and values. 
    """
    files = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
    if not files:
        print("❌ No .npy files found for verification.")
        return

    print(f"\n🔍 Verifying {num_samples} sample files from '{output_dir}':")
    samples = np.random.choice(files, min(num_samples, len(files)), replace=False)

    for f in samples:
        path = os.path.join(output_dir, f)
        arr = np.load(path)

        print(f"\nFile: {f}")
        print(f"Shape: {arr.shape}")
        print(arr)

        if arr.ndim != 2 or arr.shape[1] != 5:
            print("❌ Format error: array must be of shape (N, 5)")
        else:
            print("✅ Format OK")
#%% 
# Base directory for data
base_dir = "/home/gl188/hw/data"
ann_dir = os.path.join(base_dir, "VOC2012_train_val/VOC2012_train_val/Annotations")
output_dir = os.path.join(base_dir, "bboxes_npy")

extract_all_annotations(ann_dir, output_dir)
verify_output(output_dir, num_samples=3)
#%% 
# Base paths (adjust only if your layout changes)
BASE_DIR = "/home/gl188/hw/data"
IMG_DIR = f"{BASE_DIR}/VOC2012_train_val/VOC2012_train_val/JPEGImages"
BBOX_DIR = f"{BASE_DIR}/bboxes_npy"

# VOC2012 class names in id order (0..19)
CLASS_NAMES = [
    "person", "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike",
    "train", "bottle", "chair", "diningtable", "pottedplant",
    "sofa", "tvmonitor"
]
#%% 
import os
import numpy as np
from PIL import Image

def load_image_by_id(image_id: str):
    """ 
    Load JPEG image by VOC id (e.g., '2008_004940' -> .../JPEGImages/2008_004940.jpg). 
    Returns a PIL.Image object. 
    """
    img_path = os.path.join(IMG_DIR, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    return Image.open(img_path).convert("RGB")

def load_bboxes_by_id(image_id: str):
    """ 
    Load bbox numpy array by VOC id (e.g., '2008_004940' -> .../bboxes_npy/2008_004940.npy). 
    Returns an array of shape (N, 5): [class_id, x, y, w, h]. 
    """
    npy_path = os.path.join(BBOX_DIR, f"{image_id}.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"BBox npy not found: {npy_path}")
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"Invalid bbox shape {arr.shape}, expected (N, 5).")
    return arr
#%% 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_bboxes_on_image(img, bboxes, title: str = None, linewidth: int = 2):
    """ 
    Draw bounding boxes on an image. 
 
    Args: 
        img: PIL.Image or numpy array (H,W,3) 
        bboxes: np.ndarray of shape (N, 5) with [class_id, x, y, w, h] 
        title: optional title string 
        linewidth: rectangle border width 
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)

    if bboxes.size == 0:
        plt.show()
        return

    for cls_id, x, y, w, h in bboxes:
        # Rectangle
        rect = patches.Rectangle(
            (int(x), int(y)), int(w), int(h),
            linewidth=linewidth, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

        # Class label
        cls_id = int(cls_id)
        label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
        ax.text(
            int(x), max(int(y) - 4, 0),
            label,
            fontsize=10, color="yellow",
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none")
        )

    plt.show()

def show_example_by_id(image_id: str, linewidth: int = 2):
    """ 
    Load image + its bboxes by id and render the visualization. 
    """
    img = load_image_by_id(image_id)
    bboxes = load_bboxes_by_id(image_id)
    visualize_bboxes_on_image(
        img, bboxes,
        title=f"{image_id}  (N={bboxes.shape[0]})",
        linewidth=linewidth
    )

import random

def show_random_sample(seed: int = None, linewidth: int = 2):
    """ 
    Randomly pick an id from BBOX_DIR and show the visualization. 
    """
    if seed is not None:
        random.seed(seed)
    ids = [os.path.splitext(f)[0] for f in os.listdir(BBOX_DIR) if f.endswith(".npy")]
    if not ids:
        raise RuntimeError(f"No .npy files under {BBOX_DIR}")
    image_id = random.choice(ids)
    show_example_by_id(image_id, linewidth=linewidth)
#%% 
show_example_by_id("2008_004940")

show_random_sample(seed=42)
#%% 
import os
import numpy as np

# Base directory
BASE_DIR = "/home/gl188/hw/data"
BBOX_DIR = os.path.join(BASE_DIR, "bboxes_npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "multi_hot_labels")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VOC2012 has 20 object categories in fixed order
CLASS_NAMES = [
    "person", "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike",
    "train", "bottle", "chair", "diningtable", "pottedplant",
    "sofa", "tvmonitor"
]
NUM_CLASSES = len(CLASS_NAMES)
#%% 
def convert_bbox_to_multihot(bbox_array, num_classes=NUM_CLASSES):
    """ 
    Convert an (N,5) bbox array [class_id, x, y, w, h] 
    into a (num_classes,) multi-hot vector. 
    """
    multihot = np.zeros(num_classes, dtype=np.float32)
    if bbox_array.size == 0:
        return multihot  # empty image (no objects)
    class_ids = np.unique(bbox_array[:, 0].astype(int))
    for cid in class_ids:
        if 0 <= cid < num_classes:
            multihot[cid] = 1.0
    return multihot

def generate_multihot_labels(bbox_dir=BBOX_DIR, output_dir=OUTPUT_DIR):
    """ 
    Convert all bbox .npy files into multi-hot label .npy files. 
    Each output file has shape (20,) corresponding to VOC classes. 
    """
    npy_files = [f for f in os.listdir(bbox_dir) if f.endswith(".npy")]
    print(f"Found {len(npy_files)} bbox files in {bbox_dir}")

    for fname in npy_files:
        bbox_path = os.path.join(bbox_dir, fname)
        bbox_array = np.load(bbox_path)
        multihot = convert_bbox_to_multihot(bbox_array)
        save_path = os.path.join(output_dir, fname)
        np.save(save_path, multihot)

    print(f"✅ Saved {len(npy_files)} multi-hot label files to {output_dir}")
#%% 
# Run the conversion
generate_multihot_labels()

# Check one example
example_id = "2008_004940"
example_path = os.path.join(OUTPUT_DIR, f"{example_id}.npy")
multi_hot = np.load(example_path)

print("Example ID:", example_id)
print("Multi-hot vector:", multi_hot)
print("Active classes:", [CLASS_NAMES[i] for i, v in enumerate(multi_hot) if v == 1])
#%% 
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T

IMG_DIR = "/home/gl188/hw/data/VOC2012_train_val/VOC2012_train_val/JPEGImages"
LABEL_DIR = "/home/gl188/hw/data/multi_hot_labels"

# Basic image preprocessing and augmentation
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

class VOCDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        label_path = os.path.join(self.label_dir, f"{img_id}.npy")

        img = Image.open(img_path).convert("RGB")
        label = np.load(label_path).astype(np.float32)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)
#%% 
from sklearn.model_selection import train_test_split

# Prepare dataset and split train/val
dataset = VOCDataset(IMG_DIR, LABEL_DIR)
train_ids, val_ids = train_test_split(dataset.ids, test_size=0.2, random_state=42)

train_dataset = VOCDataset(IMG_DIR, LABEL_DIR, transform=train_transform)
val_dataset = VOCDataset(IMG_DIR, LABEL_DIR, transform=val_transform)
train_dataset.ids = train_ids
val_dataset.ids = val_ids

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16)
#%% 
import torch.nn as nn
import torchvision.models as models

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)  # replace final layer

    def forward(self, x):
        return self.backbone(x)

device = torch.device("cuda")
model = MultiLabelResNet(num_classes=20).to(device)
#%% 
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)
#%% 
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#%% 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.cpu().numpy()

        # Forward pass
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()  # convert to probabilities

        # Store
        all_preds.append(probs)
        all_labels.append(labels)

# Concatenate all batches
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

threshold = 0.5
binary_preds = (all_preds > threshold).astype(int)
binary_labels = all_labels.astype(int)


f1_micro = f1_score(binary_labels, binary_preds, average="micro")
f1_macro = f1_score(binary_labels, binary_preds, average="macro")
precision = precision_score(binary_labels, binary_preds, average="micro")
recall = recall_score(binary_labels, binary_preds, average="micro")

print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall:    {recall:.4f}")
print(f"Validation F1 (micro): {f1_micro:.4f}")
print(f"Validation F1 (macro): {f1_macro:.4f}")
