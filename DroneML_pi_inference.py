import torch
from torchvision import transforms
from PIL import Image

# REMOVED: All training-related imports (optim, DataLoader, datasets)
# WHY: Pi only does inference, removing unused imports saves memory

# CHANGED: Load full model instead of reconstructing architecture
# WHY: Simpler code, no need to define model structure again
# Just load the trained model directly from the file
spatial = torch.load(r"c:\Users\shabd\Documents\AURORA\ML\spatial_person_detector_full.pth", map_location=torch.device('cpu'), weights_only=False)

# CHANGED: Force CPU usage (removed CUDA check)
# WHY: Raspberry Pi doesn't have CUDA/GPU, always uses CPU
device = torch.device("cpu")
spatial.to(device)

# CHANGED: Set to eval mode immediately
# WHY: Pi never trains, only does inference
# eval() disables dropout and batch normalization training behavior
spatial.eval()

# CHANGED: Reduced image size to match training (128 instead of 224)
# WHY: Must match the input size the model was trained on
# Smaller size = faster processing on Pi's limited CPU
preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# REMOVED: All training code (dataset loading, training loop, optimizer, etc.)
# WHY: Pi only runs inference, training code wastes memory and storage

#---------------------INFERENCE-----------------------

# CHANGED: Wrapped in a function for reusability
# WHY: Easier to call repeatedly for real-time drone detection
def detect_person(image_path):
    """
    Detect if a person is present in the image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        prediction: "person" or "no_person"
        confidence: Probability score (0-1)
    """
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    
    # CHANGED: Explicitly move to CPU (though already on CPU)
    # WHY: Ensures compatibility, no accidental GPU calls
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # ADDED: torch.no_grad() context
    # WHY: Disables gradient calculation, saves memory and speeds up inference
    # Critical for Pi's limited resources
    with torch.no_grad():
        output = spatial(input_batch)
        
        # ADDED: Softmax to get probabilities
        # WHY: Provides confidence scores, useful for filtering uncertain detections
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    classes = ["no_person", "person"]
    prediction = classes[predicted_idx.item()]
    
    return prediction, confidence.item()

# Example usage
if __name__ == "__main__":
    # Put your test image in the same folder as this script
    # Change 'test_image.jpg' to your actual image filename
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'test_image.jpg'
    
    try:
        prediction, confidence = detect_person(image_path)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
    except FileNotFoundError:
        print(f"Error: Image '{image_path}' not found!")
        print(f"Put a test image in: c:\\Users\\shabd\\Documents\\AURORA\\ML\\Drone-Machine-Learning\\")
        print(f"Then run: python DroneML_pi_inference.py your_image.jpg")
