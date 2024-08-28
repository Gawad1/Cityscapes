import os
import pandas as pd

class CityscapesDatasetCreator:
    def __init__(self, config: dict):
        self.config = config
        self.images_dir = config['dataset_paths']['images']
        self.masks_dir = config['dataset_paths']['gtFine']
        self.test_csv = config['test_csv']
        self.root_dir = config['root_dir']

    def create_csv(self):
        """
        Creates a CSV file mapping image paths to mask paths.
        """
        # Initialize data dictionary
        data = {'image': [], 'mask': []}

        # Collect image and mask file names
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('_leftImg8bit.png')]
        mask_files = [f for f in os.listdir(self.masks_dir) if f.endswith('_gtFine_labelTrainIds.png')]

        print(f"Found {len(image_files)} images and {len(mask_files)} masks.")

        # Convert lists to sets for fast membership testing
        mask_files_set = set(mask_files)

        # Iterate through the image files
        for img_file in image_files:
            # Create corresponding mask file name
            mask_file = img_file.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')

            if mask_file in mask_files_set:
                # Append data with paths relative to root_dir
                data['image'].append(os.path.relpath(os.path.join(self.images_dir, img_file), self.root_dir))
                data['mask'].append(os.path.relpath(os.path.join(self.masks_dir, mask_file), self.root_dir))

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv, index=False)
