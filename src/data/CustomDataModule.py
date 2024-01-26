import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src.data.CustomDataset import CustomDataset
from src.utils.utils import load_config_yml


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, config_dic, batch_size=32):

        super().__init__()
        self.root_dir = root_dir

        ########################## Load config file parameters ##########################
        self.config = config_dic #load_config_yml('src/config/config.yml')
        self.resize_width =   self.config['dataset']['resize_width']
        self.resize_height =  self.config['dataset']['resize_height']
        self.train_size =     self.config['dataset']['train_size']
        self.test_size =      self.config['dataset']['test_size']
        self.valid_size =     self.config['dataset']['valid_size']
        self.num_workers =    self.config['dataset']['num_workers']
        self.batch_size =     self.config['train']['batch_size']


    def setup(self):
        dataset = CustomDataset(self.root_dir, transform=transforms.Compose([
            transforms.Resize((self.resize_width , self.resize_height)),
            transforms.ToTensor()
        ]))

        train_size = int(self.train_size * len(dataset)) # 70% Train
        test_size =  int(self.test_size * len(dataset)) # 15% Test
        valid_size = int(self.valid_size * len(dataset)) # 15% Valid
        #add the difference between dataset and (train+test+valid) to valid
        valid_size = valid_size + (len(dataset) - (train_size+test_size+valid_size)) 
        self.train_dataset, self.test_dataset, self.valid_dataset, = torch.utils.data.random_split(dataset, [train_size, test_size,valid_size])
        return self.train_dataset, self.test_dataset,self.valid_dataset, dataset.get_classes()

    def train_dataloader(self,train_dataset):
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self,val_dataset):
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self,test_dataset):
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
