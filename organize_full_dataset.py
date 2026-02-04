import shutil
from pathlib import Path

print("="*60)
print("Organizing COCO for Full Dataset")
print("="*60)

coco_path = Path(r'C:\Users\shabd\Documents\AURORA\val2017')
all_images = list(coco_path.glob('*.jpg'))

print(f"Found {len(all_images)} COCO images")

# Use 80/20 split
train_count = int(len(all_images) * 0.8)
train_images = all_images[:train_count]
val_images = all_images[train_count:]

train_dest = Path(r'C:\Users\shabd\Documents\AURORA\dataset\train\no_person')
val_dest = Path(r'C:\Users\shabd\Documents\AURORA\dataset\val\no_person')

train_dest.mkdir(parents=True, exist_ok=True)
val_dest.mkdir(parents=True, exist_ok=True)

print(f"\nCopying {len(train_images)} to training...")
for i, img in enumerate(train_images):
    shutil.copy(img, train_dest / img.name)
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(train_images)}...")

print(f"\nCopying {len(val_images)} to validation...")
for i, img in enumerate(val_images):
    shutil.copy(img, val_dest / img.name)
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(val_images)}...")

print("\n" + "="*60)
print("[OK] Full dataset ready!")
print("="*60)
print(f"Training no_person: {len(list(train_dest.glob('*.jpg')))} images")
print(f"Validation no_person: {len(list(val_dest.glob('*.jpg')))} images")
print("\nNow run: python DroneML_train.py")
