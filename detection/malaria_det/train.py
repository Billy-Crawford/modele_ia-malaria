import os
from ultralytics import YOLO

def main():
    # Define paths
    dataset_yaml = "data/processed/nlm_pf_yolo/dataset.yaml"
    
    # Check if dataset yaml exists
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found at {dataset_yaml}")
        
    print("Initializing YOLOv8 Nano model...")
    # Initialize YOLO model
    # We use YOLOv8 Nano (n) because it is highly optimized for Edge/Mobile (Flutter deployment)
    model = YOLO("yolov8n.pt") 
    
    print("Starting training...")
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=100,           # Train for up to 100 epochs
        imgsz=640,            # Image size (matches our tiling strategy)
        batch=16,             # Batch size
        project="runs/detect",
        name="malaria_yolov8n",
        exist_ok=True,
        patience=20,          # Early stopping if no improvement for 20 epochs
        save=True,            # Save checkpoints
        val=True,             # Validate during training
        plots=True            # Generate plots (confusion matrix, PR curve, etc.)
    )
    
    print("Training complete!")
    
    # Let's export to TFLite for Edge Deployment (Flutter app)
    # TFLite is standard for mobile device edge inference.
    print("Exporting model to TFLite for mobile inference...")
    try:
        # export to tflite
        model.export(format="tflite")
    except Exception as e:
        print(f"Warning: TFLite export encountered an issue: {e}")
        print("Model is still saved as .pt format.")
        
if __name__ == "__main__":
    main()
