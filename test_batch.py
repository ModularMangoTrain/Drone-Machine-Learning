import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Load model
print("Loading model...")
spatial = torch.load("spatial_person_detector_quantized.pth", map_location=torch.device('cpu'), weights_only=False)
device = torch.device("cpu")
spatial.to(device)
spatial.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def test_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.inference_mode():
        output = spatial(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    classes = ["no_person", "person"]
    return classes[predicted_idx.item()], confidence.item()

# Test person images
print("\n" + "="*60)
print("Testing PERSON images:")
print("="*60)
person_dir = Path(r'C:\Users\shabd\Documents\AURORA\dataset\train\person')
person_images = list(person_dir.glob('*.jpg'))[:5]

for img in person_images:
    pred, conf = test_image(img)
    status = "✓" if pred == "person" else "✗"
    print(f"{status} {img.name}: {pred} ({conf:.1%})")

# Test no_person images
print("\n" + "="*60)
print("Testing NO_PERSON images:")
print("="*60)
no_person_dir = Path(r'C:\Users\shabd\Documents\AURORA\dataset\train\no_person')
no_person_images = list(no_person_dir.glob('*.jpg'))[:5]

for img in no_person_images:
    pred, conf = test_image(img)
    status = "✓" if pred == "no_person" else "✗"
    print(f"{status} {img.name}: {pred} ({conf:.1%})")

print("\n" + "="*60)
print("Testing complete!")
print("="*60)
