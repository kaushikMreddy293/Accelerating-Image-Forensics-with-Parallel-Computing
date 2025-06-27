# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from transformers import ViTForImageClassification, ViTFeatureExtractor
# from PIL import Image
# import numpy as np

# class ResNet18Model:
#     def __init__(self, model_path=None):
#         """Initialize ResNet18 model for AI vs Real image detection
        
#         Args:
#             model_path: Path to the saved model weights (optional)
#         """
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Initialize model
#         self.model = models.resnet18(pretrained=True)
#         self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
#         # Load weights if provided
#         if model_path:
#             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
#         self.model = self.model.to(self.device)
#         self.model.eval()
        
#         # Define transforms
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                 [0.229, 0.224, 0.225])
#         ])
        
#         # Class labels
#         self.classes = ['Real', 'AI-generated']
    
#     def predict(self, image):
#         """Predict whether the image is real or AI-generated
        
#         Args:
#             image: PIL Image object
            
#         Returns:
#             dict: Prediction results including class, probabilities, and processing time
#         """
#         # Start timing
#         start_time = torch.cuda.Event(enable_timing=True)
#         end_time = torch.cuda.Event(enable_timing=True)
        
#         start_time.record()
        
#         # Preprocess image
#         img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
#         # Get prediction
#         with torch.no_grad():
#             outputs = self.model(img_tensor)
#             probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
#         end_time.record()
#         torch.cuda.synchronize()
        
#         # Get processing time in milliseconds
#         processing_time = start_time.elapsed_time(end_time)
        
#         # Get prediction class and confidence
#         predicted_class = int(torch.argmax(probabilities).item())
#         confidence = float(probabilities[predicted_class].item())
        
#         # Get all class probabilities
#         all_probs = {self.classes[i]: float(prob.item()) for i, prob in enumerate(probabilities)}
        
#         return {
#             'class': self.classes[predicted_class],
#             'confidence': confidence,
#             'probabilities': all_probs,
#             'processing_time_ms': processing_time
#         }


# class VisionTransformerModel:
#     def __init__(self, model_path=None):
#         """Initialize Vision Transformer model for AI vs Real image detection
        
#         Args:
#             model_path: Path to the saved model weights (optional)
#         """
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Initialize feature extractor
#         self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        
#         # Initialize model
#         self.model = ViTForImageClassification.from_pretrained(
#             'google/vit-base-patch16-224',
#             num_labels=2,
#             ignore_mismatched_sizes=True
#         ).to(self.device)
        
#         # Load weights if provided
#         if model_path:
#             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
#         self.model.eval()
        
#         # Define transforms
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=self.feature_extractor.image_mean, 
#                 std=self.feature_extractor.image_std
#             )
#         ])
        
#         # Class labels
#         self.classes = ['Real', 'AI-generated']
    
#     def predict(self, image):
#         """Predict whether the image is real or AI-generated
        
#         Args:
#             image: PIL Image object
            
#         Returns:
#             dict: Prediction results including class, probabilities, and processing time
#         """
#         # Start timing
#         start_time = torch.cuda.Event(enable_timing=True)
#         end_time = torch.cuda.Event(enable_timing=True)
        
#         start_time.record()
        
#         # Preprocess image
#         img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
#         # Get prediction
#         with torch.no_grad():
#             outputs = self.model(img_tensor).logits
#             probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
#         end_time.record()
#         torch.cuda.synchronize()
        
#         # Get processing time in milliseconds
#         processing_time = start_time.elapsed_time(end_time)
        
#         # Get prediction class and confidence
#         predicted_class = int(torch.argmax(probabilities).item())
#         confidence = float(probabilities[predicted_class].item())
        
#         # Get all class probabilities
#         all_probs = {self.classes[i]: float(prob.item()) for i, prob in enumerate(probabilities)}
        
#         return {
#             'class': self.classes[predicted_class],
#             'confidence': confidence,
#             'probabilities': all_probs,
#             'processing_time_ms': processing_time
#         }

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import time

class ResNet18Model:
    def __init__(self, model_path=None):
        """Initialize ResNet18 model for AI vs Real image detection
        
        Args:
            model_path: Path to the saved model weights (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = models.resnet18(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        # Load weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        
        # Class labels
        self.classes = ['Real', 'AI-generated']
    
    def predict(self, image):
        """Predict whether the image is real or AI-generated
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Prediction results including class, probabilities, and processing time
        """
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # End timing
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Get prediction class and confidence
        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())
        
        # Get all class probabilities
        all_probs = {self.classes[i]: float(prob.item()) for i, prob in enumerate(probabilities)}
        
        return {
            'class': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': all_probs,
            'processing_time_ms': processing_time
        }


class VisionTransformerModel:
    def __init__(self, model_path=None):
        """Initialize Vision Transformer model for AI vs Real image detection
        
        Args:
            model_path: Path to the saved model weights (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize feature extractor
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        
        # Initialize model
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Load weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.feature_extractor.image_mean, 
                std=self.feature_extractor.image_std
            )
        ])
        
        # Class labels
        self.classes = ['Real', 'AI-generated']
    
    def predict(self, image):
        """Predict whether the image is real or AI-generated
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Prediction results including class, probabilities, and processing time
        """
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(img_tensor).logits
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # End timing
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Get prediction class and confidence
        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())
        
        # Get all class probabilities
        all_probs = {self.classes[i]: float(prob.item()) for i, prob in enumerate(probabilities)}
        
        return {
            'class': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': all_probs,
            'processing_time_ms': processing_time
        }
