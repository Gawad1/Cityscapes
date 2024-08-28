import cv2
import numpy as np

class LoadImage:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key == 'image':
                sample[key] = cv2.imread(sample[key], cv2.IMREAD_COLOR)
            elif key == 'mask':
                sample[key] = cv2.imread(sample[key], cv2.IMREAD_GRAYSCALE)

            # Check if the image was loaded correctly
            if sample[key] is None:
                raise ValueError(f"Failed to load image or mask: {sample[key]}")

        return sample

class ResizeImages:
    def __init__(self, size):
        self.size = size  # size should be (height, width)

    def __call__(self, sample):
        # Resize the image with bilinear interpolation
        if 'image' in sample:
            sample['image'] = cv2.resize(sample['image'], (self.size[1], self.size[0]),
                                         interpolation=cv2.INTER_LINEAR)

        # Resize the mask with nearest-neighbor interpolation
        if 'mask' in sample:
            sample['mask'] = cv2.resize(sample['mask'], (self.size[1], self.size[0]),
                                        interpolation=cv2.INTER_NEAREST)

        return sample

class MapLabels:
    def __init__(self, keys, ignore_label_id=255):
        self.keys = keys
        self.ignore_label_id = ignore_label_id

    def __call__(self, sample):
        for key in self.keys:
            mask = sample[key]

            # Initialize the mapping array
            mapping = np.arange(256)  # Create a mapping from 0 to 255
            mapping[self.ignore_label_id] = 0  # Map 255 to 0
            mapping[0:20] = np.arange(1, 21)  # Map 0-19 to 1-20

            # Apply the mapping to the mask
            sample[key] = mapping[mask]
        return sample


class BlendCLAHEandNormalize:
    def __init__(self, keys, mean, std, alpha=0.5, clip_limit=1.5, tile_grid_size=(8, 8)):
        self.keys = keys
        self.mean = mean
        self.std = std
        self.alpha = alpha
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, sample):
        for key in self.keys:
            img = sample[key]
            if len(img.shape) == 2:  # Grayscale image
                img_clahe = self.clahe.apply(img)
            else:  # Color image, apply CLAHE on each channel separately
                img_clahe = np.stack([self.clahe.apply(channel) for channel in cv2.split(img)], axis=-1)

            img_clahe = img_clahe.astype(np.float32) / 255.0
            img_normalized = (img_clahe - self.mean) / self.std

            # Blend CLAHE-enhanced image with normalized image
            sample[key] = self.alpha * img_clahe + (1 - self.alpha) * img_normalized

        return sample

