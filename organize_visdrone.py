import shutil
from pathlib import Path
import random

print("="*60)
print("Filtering VisDrone for No-Person Images")
print("="*60)

# VisDrone paths
visdrone_images = Path(r'C:\Users\shabd\Documents\AURORA\VisDrone2019-DET-train\images')
visdrone_annotations = Path(r'C:\Users\shabd\Documents\AURORA\VisDrone2019-DET-train\annotations')

if not visdrone_images.exists():
    print(f"ERROR: {visdrone_images} not found!")
    print("Download VisDrone and extract to C:\\Users\\shabd\\Documents\\AURORA\\VisDrone")
    exit(1)

# Get all images
all_images = list(visdrone_images.glob('*.jpg'))
print(f"Found {len(all_images)} VisDrone images")

# Filter images with no people (class 1 = pedestrian in VisDrone)
no_person_images = []

for img in all_images:
    annotation_file = visdrone_annotations / f"{img.stem}.txt"
    
    if not annotation_file.exists():
        no_person_images.append(img)
        continue
    
    # Read annotations
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    
    # Check if any annotation has class 1 (pedestrian)
    has_person = False
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            class_id = int(parts[5])
            if class_id == 1:  # pedestrian
                has_person = True
                break
    
    if not has_person:
        no_person_images.append(img)

print(f"Found {len(no_person_images)} images with no people")

if len(no_person_images) < 100:
    print("WARNING: Very few no-person images found. Check VisDrone structure.")
    exit(1)

# Shuffle and split
random.seed(42)
random.shuffle(no_person_images)

train_count = int(len(no_person_images) * 0.8)
train_images = no_person_images[:train_count]
val_images = no_person_images[train_count:]

# Destination folders
train_dest = Path(r'C:\Users\shabd\Documents\AURORA\dataset\train\no_person')
val_dest = Path(r'C:\Users\shabd\Documents\AURORA\dataset\val\no_person')

# Clear existing no_person images
print("\nClearing old no_person images...")
for f in train_dest.glob('*.jpg'):
    f.unlink()
for f in val_dest.glob('*.jpg'):
    f.unlink()

train_dest.mkdir(parents=True, exist_ok=True)
val_dest.mkdir(parents=True, exist_ok=True)

# Copy images
print(f"\nCopying {len(train_images)} to training...")
for i, img in enumerate(train_images):
    shutil.copy(img, train_dest / img.name)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(train_images)}...")

print(f"\nCopying {len(val_images)} to validation...")
for i, img in enumerate(val_images):
    shutil.copy(img, val_dest / img.name)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(val_images)}...")

print("\n" + "="*60)
print("[OK] VisDrone no-person dataset ready!")
print("="*60)
print(f"Training no_person: {len(train_images)} images")
print(f"Validation no_person: {len(val_images)} images")
print("\nNow run: python DroneML_train.py")
