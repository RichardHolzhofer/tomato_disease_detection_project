import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import dagshub
import mlflow

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.20090248994381077, inplace=True),
            nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)




class ImageClassifier():
    def __init__(self, model_path, class_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomEfficientNet(num_classes=11)
        
        # 1. Attempt to load the Production model from DagsHub MLflow publicly
        try:
            print("Attempting to connect to DagsHub MLflow to fetch the Production model...")
            
            # Set the tracking URI explicitly to the public DagsHub repo instead of using dagshub.init()
            # This allows unauthenticated users to download public model artifacts.
            mlflow.set_tracking_uri("https://dagshub.com/RichardHolzhofer/tomato_disease_detection_project.mlflow")
            
            # The model was registered under the filename during training
            model_name = "efficientnet_b0_ff.pth"
            model_uri = f"models:/{model_name}/Production"
            
            # Load the model directly from the MLflow artifact
            loaded_model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            # Transfer the state dict to our local architecture
            self.model.efficientnet.load_state_dict(loaded_model.state_dict())
            print("✅ Successfully loaded model from MLflow registry.")
            
        except Exception as e:
            # 2. Fallback to local model_path if DagsHub connection fails
            print(f"⚠️ Failed to load from MLflow: {e}. Falling back to local model at {model_path}.")
            self.model.efficientnet.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model.to(self.device)
        self.model.eval()   

        if class_name is None:
            self.class_name = {
                0: 'Bacterial spot',
                1: 'Early blight',
                2: 'Late blight',
                3: 'Leaf Mold',
                4: 'Septoria leaf spot',
                5: 'Spider mites, Two-spotted spider mite',
                6: 'Target Spot',
                7: 'Tomato Yellow Leaf Curl Virus',
                8: 'Tomato mosaic virus',
                9: 'Healthy',
                10: 'Powdery mildew'
                }
        else:
            self.class_name = class_name

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(image_tensor)
            prediction = output.argmax(dim=1)
            
        label = self.class_name[prediction.item()]

        cwd = os.getcwd()
        output_path = os.path.join(cwd, "output.png")

        fig, ax = plt.subplots()
        plt.imshow(image)
        
        if label == "Healthy":
            plt.title(f"No disease detected, your plant is {label.lower()}")
        else:
            plt.title(f"Detected disease: {label.lower()}")
        plt.axis("off")
        plt.savefig(output_path)

        return label, output_path