import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import os
import sys

from torchvision import datasets, transforms
from torchvision.io import read_image
from timm import create_model
from timm.data import resolve_data_config, create_transform
from pytorch_lightning.callbacks import LearningRateMonitor
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy
from PIL import Image

from src.data.CustomDataset import CustomDataset
from src.data.CustomDataModule import CustomDataModule
from src.model.CustomClassifier import CustomClassifier
from src.inference.inference import *
from src.utils.utils import load_config_yml,extract_targets,tensors_to_img

from datetime import datetime


if __name__ == "__main__":

    ########################## Load config file parameters ##########################
    base_root = os.path.dirname(os.path.abspath(__file__))
    directory='src/config/'
    config_files = [f for f in os.listdir(directory) if f.endswith(".yml") or f.endswith(".yaml")]
    print(f"List config file: {config_files}")

    for file_name in config_files:
        
        config_path = os.path.join(directory, file_name)
        config = load_config_yml(config_path)
        print(f'\n [RUNNING] config file: {file_name} \n')

        #config_path = 'src/config/config.yml'
        #config = load_config_yml(config_path)

        report_folder =     config['report']['folder']
        data_folder =       config['dataset']['folder']
        epochs =            config['train']['epochs']
        batch_size =        config['train']['batch_size']
        folder_pth =        config['train']['folder_pth']
        checkpoint_path =   config['evaluate']['checkpoint_path']

        data_folder =       base_root + data_folder
        folder_pth =        base_root + folder_pth
        checkpoint_path =   base_root + checkpoint_path
        dataset_name =      data_folder.split('/')[-1]
        config_file_name =  file_name.replace('.yml', '')#config_path.split('/')[-1].replace('.yml','')

        report_name = 'report_' + config_file_name + '.txt'
        model_name = 'model_' + dataset_name + '_' + config_file_name + '.pth'

        model_folder = folder_pth + model_name

        # if you will run the same config file many times the system will write on the same report
        report_folder = base_root + report_folder + report_name
        
        # The first parameter (sys.argv[0]) is the script name
        mode = sys.argv[1]
        with open(report_folder, 'a') as file:
            print("###################################################", file=file)
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DATETIME]: {formatted_datetime}", file=file)
            print(f"    [BASE_ROOT] {base_root}", file=file)
            print(f"    [MODE]: {mode}", file=file) # the mode could be train/test/both
            print(f"    [DATA_FOLDER]: {data_folder}", file=file) 
            print(f"    [MODEL_FOLDER]: {model_folder}", file=file) 
            print(f"    [DATASET]: {dataset_name}", file=file)
            print(f"    [CONFIG FILE]: {config_file_name}", file=file)
            print(f"    [REPORT FOLDER]: {report_folder}", file=file)


        ################################## Load Dataset ##################################

        data_module = CustomDataModule(data_folder, config_dic = config)

        train_dataset = data_module.setup()[0]
        test_dataset =  data_module.setup()[1]
        val_dataset =  data_module.setup()[2]
        classes_list =  data_module.setup()[3]
        num_classes = len(classes_list)

        train_dataloader = data_module.train_dataloader(train_dataset)
        for batch in train_dataloader:
            inputs_tr, targets = batch
            break
        
        test_dataloader = data_module.test_dataloader(test_dataset)
        for batch in test_dataloader:
            inputs_te, targets = batch
            break

        val_dataloader = data_module.val_dataloader(val_dataset)
        for batch in val_dataloader:
            inputs_v, targets = batch
            break

        train_loader = train_dataloader
        val_loader =   val_dataloader
        test_loader =  test_dataloader
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(report_folder, 'a') as file:
            print(f"    [DEVICE]: {device}", file=file) 
            print(f"    [CLASSES LIST]: {classes_list}", file=file) 
            print(f"    [CLASSES NUM]: {num_classes}", file=file) 
            print(f"    [NUM TRAINLOADER ELEMENTS]: {len(train_dataloader)*batch_size}", file=file) 
            print(f"    [SINGLE TRAINLOADER SHAPE]: {inputs_tr.shape}", file=file) 
            print(f"    [NUM TESTLOADER ELEMENTS]: {len(test_dataloader)*batch_size}", file=file) 
            print(f"    [SINGLE TESTLOADER SHAPE]: {inputs_te.shape}", file=file) 
            print(f"    [NUM VALIDLOADER ELEMENTS]: {len(val_dataloader)*batch_size}", file=file) 
            print(f"    [SINGLE VALIDLOADER SHAPE]: {inputs_v.shape}", file=file) 

        ep_str =        '[EPOCHS]: XXXXX'
        start_tr_str =  '[START TRAINING]: XXXXX'
        end_tr_str =    '[END TRAINING]: XXXXX'
        start_te_str =  '[START TEST]: XXXXX'
        end_te_str =    '[END TEST]: XXXXX'
        acc_str =       '[ACCURACY]: XXXXX'

        if mode == 'train':    #################################### Training ####################################
            
            ep_str = f"[EPOCHS]: {epochs}"
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            start_tr_str =  f"[START TRAINING]: {formatted_datetime}"
            model = CustomClassifier(num_classes=num_classes,base_root = base_root, config_dic = config)
            lr_monitor = LearningRateMonitor(logging_interval='step')
            trainer = pl.Trainer(accelerator = device, devices = 1, min_epochs=0, max_epochs=epochs, check_val_every_n_epoch=1,val_check_interval=0.5, callbacks=[lr_monitor])
            trainer.fit(model, train_loader, val_loader)
            #torch.save(model.state_dict(), model_folder)
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            end_tr_str = f"[END TRAINING]: {formatted_datetime}"

        elif mode == 'both':   ############################## Training + Inference ##############################
            
            ep_str = f"[EPOCHS]: {epochs}"
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            start_tr_str =  f"[START TRAINING]: {formatted_datetime}"

            model = CustomClassifier(num_classes=num_classes,base_root = base_root, config_dic = config)
            lr_monitor = LearningRateMonitor(logging_interval='step')
            trainer = pl.Trainer(accelerator = device, devices = 1, min_epochs=0, max_epochs=epochs, check_val_every_n_epoch=1,val_check_interval=0.5, callbacks=[lr_monitor])
            trainer.fit(model, train_loader, val_loader)
            #torch.save(model.state_dict(), model_folder)
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            end_tr_str = f"[END TRAINING]: {formatted_datetime}"
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            start_te_str =  f"[START TEST]: {formatted_datetime}"

            predictions = infer(model, val_loader)
            real_targets = extract_targets(val_loader)
            #Accuracy with valid
            acc = 0
            total = len(predictions)
            for el in range(0,total):
                if predictions[el] - real_targets[el] == 0:
                    acc = acc + 1
            acc = acc / total
            
            acc_str = f"[ACCURACY]: {acc}" 
            #tensors_to_img(test_dataloader,5)      
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            end_te_str = f"[END TEST]: {formatted_datetime}"

        elif mode == 'test':   #################################### Inference ####################################

            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            start_te_str =  f"[START TEST]: {formatted_datetime}"

            model = CustomClassifier.load_from_checkpoint(num_classes=num_classes,checkpoint_path=checkpoint_path, base_root=base_root, config_dic = config)
            predictions = infer(model, val_loader)
            real_targets = extract_targets(val_loader)

            #Accuracy with test
            acc = 0
            total = len(predictions)
            for el in range(0,total):
                if predictions[el] - real_targets[el] == 0:
                    acc = acc + 1
            acc = acc / total
            acc_str = f"[ACCURACY]: {acc}" 
            #tensors_to_img(test_dataloader,5)

            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            end_te_str = f"[END TEST]: {formatted_datetime}"
        else:                   ##################################### ERROR #####################################
            print("[ERROR] run from terminal python main.py parameter_1") 
        

        with open(report_folder, 'a') as file:
            print(f"    {ep_str}", file=file) 
            print(f"    {start_tr_str}", file=file) 
            print(f"    {end_tr_str}", file=file) 
            print(f"    {start_te_str}", file=file) 
            print(f"    {end_te_str}", file=file) 
            print(f"    {acc_str}", file=file) 
