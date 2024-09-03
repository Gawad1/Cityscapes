from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import matplotlib.pyplot as plt  # Import matplotlib for colormap
from src.models import UNet
from src.transforms import ResizeImages, LoadImage, BlendCLAHEandNormalize
from src.dataset import CityscapesDataset

app = Flask(__name__)

# Directory for output images
OUTPUT_FOLDER = '/app/output'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the model and set it to evaluation mode
model = UNet(in_channels=3, out_channels=20)  # Adjust the number of output channels if necessary
checkpoint = torch.load('/app/model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Mean and standard deviation for normalization
mean = [0.28363052, 0.32439385, 0.28523327]
std_dev = [0.18928334, 0.19246128, 0.18998726]

def preprocess_image(image_path):
    # Define transformations
    transform = Compose([
        LoadImage(keys=['image']),
        ResizeImages(size=(256, 256)),  # Adjust size as needed
        BlendCLAHEandNormalize(keys=['image'], mean=mean, std=std_dev)
    ])

    # Create a DataFrame for the image path
    test_df = pd.DataFrame({'image': [image_path]})

    # Create the dataset
    dataset = CityscapesDataset(input_dataframe=test_df, root_dir='', KeysOfInterest=['image'], data_transform=transform)
    dl_test = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get the transformed image
    image = next(iter(dl_test))['image']
    image = image.permute(0, 3, 1, 2).to(dtype=torch.float32)
    return image

def predict_mask(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.squeeze().cpu().numpy()

def resize_mask_pil(mask, new_size):
    # Convert mask to PIL Image
    mask_image = Image.fromarray(mask.astype(np.uint8))

    # Resize the mask using nearest-neighbor interpolation
    resized_mask_image = mask_image.resize(new_size, Image.NEAREST)

    # Convert back to numpy array
    resized_mask = np.array(resized_mask_image)

    return resized_mask

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the uploaded image
        filename = 'image.png'
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        image = preprocess_image(file_path)

        # Predict the mask
        mask = predict_mask(image)
        mask=np.array(mask)
        def resize_mask_pil(mask, new_size):
            # Convert mask to PIL Image
            mask_image = Image.fromarray(mask.astype(np.uint8))

            # Resize the mask using nearest-neighbor interpolation
            resized_mask_image = mask_image.resize(new_size, Image.NEAREST)

            # Convert back to numpy array
            resized_mask = np.array(resized_mask_image)

            return resized_mask
        # Save the mask with colormap
        new_size = (2048, 1024)  # PIL expects (width, height)
        resized_mask = resize_mask_pil(mask, new_size)
        mask_filename = 'predicted.png'
        mask_path = os.path.join(app.config['OUTPUT_FOLDER'], mask_filename)
        plt.imsave(mask_path, resized_mask, cmap='tab20', vmin=0, vmax=19)  # Save mask with colormap

        return render_template('index.html', image_url=url_for('serve_image', filename='image.png'),
                               mask_url=url_for('serve_image', filename='predicted.png'))

@app.route('/serve_image/<filename>')
def serve_image(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
