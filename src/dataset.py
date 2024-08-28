from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List
from torchvision.transforms import Compose


class CityscapesDataset(Dataset):
    def __init__(self, input_dataframe: pd.DataFrame, root_dir: str, KeysOfInterest: List[str],
                 data_transform: Compose):
        self.root_dir = root_dir
        self.koi = KeysOfInterest
        self.input_dataframe = input_dataframe[self.koi]
        self.data_transform = data_transform

    def __getitem__(self, item: int):
        # Get the row from the DataFrame
        row = self.input_dataframe.iloc[item]

        # Construct the file paths
        sample = {}
        for key in self.koi:
            file_path = os.path.join(self.root_dir, row[key])
            sample[key] = file_path

        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

    def __len__(self):
        return len(self.input_dataframe)