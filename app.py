import os
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Path untuk menyimpan gambar yang diupload
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Define the TinyCNN model using Depthwise Separable Convolutions for Alzheimer's MRI
# Define a Depthwise Separable Convolutional Layer
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        # Use Depthwise Separable Convolutions for efficient computation
        self.conv1 = DepthwiseSeparableConv(1, 16, kernel_size=3, stride=1, padding=1)  # 1 input channel, 16 output channels
        self.conv2 = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1, padding=1)  # 16 input channels, 32 output channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer with 2 output classes for binary classification
        self.fc1 = nn.Linear(32 * 7 * 7, 2)  # Output layer for 2 classes (0: Non-Demented, 1: Demented)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = x.view(-1, 32 * 7 * 7)  # Flatten for fully connected layer
        x = self.fc1(x)  # Output layer
        return x

# Correct way: Load the model architecture first, then the state dict
model = TinyCNN()  # Replace YourModelClass with your actual model class
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set the model to evaluation mode


# Dummy class names
class_names = ["Healthy", "Demented"]

# Preprocessing untuk gambar
def preprocess_image(image_path):
    input_image = Image.open(image_path).convert("L")  # Step 1: Convert to grayscale (1 channel)

    # Step 2: Increase contrast to highlight brain structures (optional)
    enhancer = ImageEnhance.Contrast(input_image)
    input_image = enhancer.enhance(1.5)  # Adjust contrast level if needed

    # Adjust coordinates (Healthy MRI)
    x, y, w, h = 0, 0, 315, 408
    cropped_image = input_image.crop((x, y, x + w, y + h))

    # Resize and normalize
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to model input dimensions
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values, adjust mean/std if needed
    ])

    input_tensor = preprocess(cropped_image).unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
    return input_tensor, cropped_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Menerima gambar yang diupload
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess image dan prediksi menggunakan model
            input_tensor, cropped_image = preprocess_image(file_path)

            # Move the model to the correct device (e.g., CUDA or CPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = input_tensor.to(device)

            # Model evaluation
            with torch.no_grad():
                output = model(input_tensor)  # Forward pass
                probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
                top_probs, top_indices = torch.topk(probabilities, 2)  # Get top 2 probabilities and indices

            # Convert probabilities to percentage format
            top_probs = top_probs.squeeze().cpu().numpy() * 100
            top_indices = top_indices.squeeze().cpu().numpy()

            # Convert the image to base64 for displaying in HTML
            # You can use io.BytesIO() to convert image to base64, or save the image to static

            result = {
                "image_path": file_path,
                "predicted_label": class_names[top_indices[0]],
                "probability": top_probs[0],
                "top_predictions": [(class_names[top_indices[i]], top_probs[i]) for i in range(2)]
            }

            return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
