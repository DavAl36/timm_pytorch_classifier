import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from timm.data import resolve_data_config, create_transform
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torchmetrics
from torchmetrics.classification import Accuracy
from src.utils.utils import load_config_yml
from torchmetrics.classification import F1Score
from sklearn.metrics import f1_score


class CustomClassifier(pl.LightningModule):

    #def __init__(self, num_classes, learning_rate=0.001):
    def __init__(self,num_classes,base_root, config_dic):

        super(CustomClassifier, self).__init__()

        ########################## Load config file parameters ##########################
        self.config = config_dic #load_config_yml('src/config/config.yml')
        self.model_name =      self.config['train']['model_name']
        self.base_model_path = self.config['train']['base_model_path']
        self.learning_rate =   self.config['train']['learning_rate']
        ##################################### Init #####################################
        self.num_classes = num_classes

        self.model = create_model(self.model_name, pretrained=False)
        self.base_root = base_root
        self.model.load_state_dict(torch.load(self.base_root +self.base_model_path+self.model_name+'.pth'))

        #self.model = create_model(self.model_name, pretrained=True)
        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1 = F1Score(num_classes=self.num_classes, average='micro',task="multiclass") 

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy(logits, y),on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log('on_train_epoch_end', self.train_accuracy.compute())

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss,on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc_step', self.val_accuracy(logits, y),on_step=True, on_epoch=True, prog_bar=True)
        f1 = self.f1(logits, y)
        self.log('val_f1', f1, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
