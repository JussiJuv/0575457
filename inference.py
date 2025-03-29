import os
import torch
from torch.utils.data import DataLoader
from model import FusionModel
from data_processing import FusionDataset
from config import *
from utils import load_checkpoint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print("Loading model...")
model = FusionModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)

# Load latest checkpoint
checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) 
                    if f.startswith("epoch_") and f.endswith(".pt")]
latest_checkpoint = max(checkpoint_files, 
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)

_, skeleton_mean, skeleton_std = load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None)
model.eval()
print(f'Model {latest_checkpoint} loaded!')

# Prepare test dataset
print("Loading test dataset...")
test_dataset = FusionDataset(split='test', num_frames=NUM_FRAMES)
test_dataset.skeleton_mean = skeleton_mean
test_dataset.skeleton_std = skeleton_std
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Loaded {len(test_dataset)} test samples")

top1_correct = 0
top5_correct = 0
total = 0
all_labels = []
all_preds = []

print("Evaluation in progress...")
with torch.no_grad():
    for rgb_inputs, skeleton_inputs, labels in test_loader:
        rgb_inputs = rgb_inputs.to(device)
        skeleton_inputs = skeleton_inputs.float().to(device)
        labels = labels.to(device)
        
        outputs = model(rgb_inputs, skeleton_inputs)
        _, preds = torch.topk(outputs, k=5, dim=1)
        
        # Store predictions and labels
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds[:, 0].cpu().numpy())
        
        # Calculate accuracy
        correct = preds.eq(labels.view(-1, 1).expand_as(preds))
        top1_correct += correct[:, 0].sum().item()
        top5_correct += correct[:, :5].sum().item()
        total += labels.size(0)

# Print basic metrics
print(f"\nTest Results:")
print(f"Top-1 Accuracy: {100 * top1_correct / total:.2f}%")
print(f"Top-5 Accuracy: {100 * top5_correct / total:.2f}%")