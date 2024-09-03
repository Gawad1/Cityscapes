from PIL import Image
import argparse
import os
import pandas as pd
from torchvision.transforms import Compose
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from labels import labels
from dataset_creator import CityscapesDatasetCreator
from transforms import ResizeImages, MapLabels, LoadImage, BlendCLAHEandNormalize
from dataset import CityscapesDataset
from models import UNet
from losses import CombinedLoss, DiceLoss
from metrics import calculate_iou

def parse_args():
    parser = argparse.ArgumentParser(description='Run segmentation model.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for dataset paths.')
    parser.add_argument('--gtFine', type=str, required=True, help='Relative path to gtFine train directory.')
    parser.add_argument('--images', type=str, required=True, help='Relative path to images train directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output prediction masks.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.pth file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Update config with arguments
    config = {
        'dataset_paths': {
            'gtFine': os.path.join(args.root_dir, args.gtFine),
            'images': os.path.join(args.root_dir, args.images)
        },
        'root_dir': args.root_dir,
        'output_dir': args.output_dir,
        'num_classes': 20,
        'input_size': (256, 256)
    }

    print(f"Using config: {config}")

    # Create the CSV if it doesn't exist (Optional if you need to create it)
    dataset_creator = CityscapesDatasetCreator(config)
    dataset_creator.create_csv()  # Comment out if CSV already exists

    # Read the dataset CSV
    test_df = pd.read_csv(os.path.join(config['output_dir'], 'test_data.csv'))  # Adjust if needed

    mean = [0.28363052, 0.32439385, 0.28523327]
    std_dev = [0.18928334, 0.19246128, 0.18998726]

    test_data_transform = Compose([
        LoadImage(keys=['image', 'mask']),
        ResizeImages(size=config['input_size']),
        MapLabels(keys=['mask']),
        BlendCLAHEandNormalize(keys=['image'], mean=mean, std=std_dev)
    ])

    # Create the dataset
    dataset = CityscapesDataset(input_dataframe=test_df, root_dir=args.root_dir, KeysOfInterest=['image', 'mask'], data_transform=test_data_transform)

    # Load data with DataLoader
    dl_test = DataLoader(dataset, batch_size=4, shuffle=False)

    # Init the model
    model = UNet(in_channels=3, out_channels=config['num_classes'])
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Prepare for evaluation
    model.eval()
    val_loss = 0.0
    val_correct_pixels = 0
    val_total_pixels = 0
    num_classes = config['num_classes']

    all_preds = []
    all_labels = []

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dl_test), desc='Evaluating', unit='batch'):
            images = batch['image']
            masks = batch['mask']
            images = images.permute(0, 3, 1, 2).to(dtype=torch.float32)
            masks = masks.to(dtype=torch.long)

            outputs = model(images)
            loss = DiceLoss(smooth=1e-6, gamma=2)(outputs, masks)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total_pixels += masks.numel()
            val_correct_pixels += (predicted == masks).sum().item()

            # Collect predictions and labels for IoU calculation
            all_preds.append(predicted.cpu())
            all_labels.append(masks.cpu())

            # Save prediction masks
            for i, pred in enumerate(predicted):
                pred_image = pred.cpu().numpy()
                pred_image_filename = os.path.join(config['output_dir'], f'pred_{batch_idx * 4 + i}.png')
                # Save each prediction as an image (adjust this depending on your data format)
                pred_image = (pred_image * 255 / num_classes).astype('uint8')
                Image.fromarray(pred_image.squeeze(), mode='L').save(pred_image_filename)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate IoU
    mean_iou, iou_list = calculate_iou(all_preds, all_labels, num_classes, ignore_index=0)
    print(f"Mean IoU: {mean_iou:.4f}")

    # Calculate and print average validation loss and accuracy
    val_loss = val_loss / len(dl_test.dataset)
    val_accuracy = 100.0 * val_correct_pixels / val_total_pixels
    print(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main()
