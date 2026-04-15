"""
evaluate.py — Generates confusion matrix, per-class metrics, and Grad-CAM visualisations.

Run after training:
    python evaluate.py
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
WEIGHTS  = 'models/waste_classifier.pth'

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_ds = ImageFolder('dataset/test', transform=transform)
loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

model = load_model()

# ── Collect predictions ───────────────────────────────────────────────────────
all_labels, all_preds = [], []
with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Classification Report ─────────────────────────────────────────────────────
print("\n── Classification Report ──")
print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=CLASSES, yticklabels=CLASSES)
axes[0].set_title('Confusion Matrix (counts)', fontsize=14)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=CLASSES, yticklabels=CLASSES)
axes[1].set_title('Confusion Matrix (normalised)', fontsize=14)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print("Saved confusion_matrix.png")

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()


def visualize_gradcam(image_path, save_path):
    if not os.path.exists(image_path):
        print(f"Skipping GradCAM — {image_path} not found")
        return

    gradcam = GradCAM(model, model.features[-1])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    outputs = model(tensor)
    class_idx = outputs.argmax().item()
    cam = gradcam.generate(tensor, class_idx)

    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR)) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(image)
    axes[1].imshow(cam_resized, alpha=0.5, cmap='jet')
    axes[1].set_title(f'Grad-CAM — Predicted: {CLASSES[class_idx]}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


# Try to visualise one sample from each class
for cls in CLASSES:
    sample_dir = f'dataset/test/{cls}'
    if os.path.isdir(sample_dir):
        samples = os.listdir(sample_dir)
        if samples:
            img_path = os.path.join(sample_dir, samples[0])
            visualize_gradcam(img_path, f'models/gradcam_{cls}.png')

print("\nEvaluation complete.")