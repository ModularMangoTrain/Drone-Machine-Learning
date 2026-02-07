import torch
from torchvision import transforms
from PIL import Image
import cv2
import time

# REMOVED: All training-related imports (optim, DataLoader, datasets)
# WHY: Pi only does inference, removing unused imports saves memory

# CHANGED: Load full model instead of reconstructing architecture
# WHY: Simpler code, no need to define model structure again
# Just load the trained model directly from the file
spatial = torch.load("spatial_person_detector_full.pth", map_location=torch.device('cpu'), weights_only=False)

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
def detect_person(image):
    """
    Detect if a person is present in the image.
    
    Args:
        image: PIL Image or numpy array (from camera)
        
    Returns:
        prediction: "person" or "no_person"
        confidence: Probability score (0-1)
    """
    with torch.inference_mode():
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        output = spatial(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    classes = ["no_person", "person"]
    prediction = classes[predicted_idx.item()]
    
    return prediction, confidence.item()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit(1)
    
    print("Camera opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        prediction, confidence = detect_person(frame)
        
        color = (0, 255, 0) if prediction == "person" else (0, 0, 255)
        text = f"{prediction}: {confidence:.2%}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Person Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
