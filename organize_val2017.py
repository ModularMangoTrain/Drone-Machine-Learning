import shutil
from pathlib import Path

"""
Organize existing COCO val2017 images into dataset
"""

print("="*60)
print("Organizing COCO val2017 Images")
print("="*60)

# Path to COCO images
coco_path = Path(r'C:\Users\shabd\Documents\AURORA\val2017')

if not coco_path.exists():
    print(f"[ERROR] Path not found: {coco_path}")
    exit(1)

# Get all images
all_images = list(coco_path.glob('*.jpg'))
print(f"\nFound {len(all_images)} images in val2017")

if len(all_images) == 0:
    print("[ERROR] No images found!")
    exit(1)

# Take 1000 for training, 300 for validation
train_images = all_images[:1000]
val_images = all_images[1000:1300]

# Destination folders
train_dest = Path(r'C:\Users\shabd\Documents\AURORA\dataset_small\train\no_person')
val_dest = Path(r'C:\Users\shabd\Documents\AURORA\dataset_small\val\no_person')

train_dest.mkdir(parents=True, exist_ok=True)
val_dest.mkdir(parents=True, exist_ok=True)

print(f"\nCopying {len(train_images)} images to training...")
for i, img in enumerate(train_images):
    shutil.copy(img, train_dest / img.name)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(train_images)}...")

print(f"\nCopying {len(val_images)} images to validation...")
for i, img in enumerate(val_images):
    shutil.copy(img, val_dest / img.name)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(val_images)}...")

print("\n" + "="*60)
print("[OK] Images organized!")
print("="*60)
print(f"Training no_person: {len(train_images)} images")
print(f"Validation no_person: {len(val_images)} images")
print("Start training: python DroneML_train.py")
