import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

from PIL import Image

class PhenoDataset(Dataset):
    def __init__(self,image_folder,ground_truth_folder=None,is_training=False):
        super(PhenoDataset,self).__init__()
        self.image_folder = image_folder
        self.training = is_training
        self.file_list = []
        for file in os.listdir(self.image_folder):
            if file.endswith(".jpg"):
                self.file_list.append(file)

    def __len__(self):
        ''' Return the number off images'''
        return len(self.file_list)

    def __getitem__(self, idx):
        ''' return idx of the dataset'''
        image = self.file_list[idx]
        return np.asarray(Image.open(os.path.join(self.image_folder,image)))

def return_pheno_dataloader(image_folder,ground_truth_folder=None,is_training=False,batch_size=256):
    pheno_dataset = PhenoDataset(image_folder,ground_truth_folder,is_training)
    pheno_dataloader = DataLoader(pheno_dataset,batch_size,shuffle=True if is_training else False)
    return pheno_dataloader

if __name__ == "__main__":
    train_folder = "/Users/viveksrivastav/Desktop/image/Cam41"
    pheno_dataset = PhenoDataset(train_folder)
    #pheno_dataloader = return_pheno_dataloader(train_folder)
    print(pheno_dataset[10].shape)

