# timm_pytorch_classifier

This work is a code base to realize AI Classification Models using Pytorch Lightning + Timm library + Tensorboard. The idea is to share an easy a plug and play code. By modifying the configuration files you can quickly try the models that the Timm library makes available.

#### How to run it

Install libraries from requirements file

```shell
pip install -r env/requirements.txt
```

Add a dateset with the following structure

```lua
Dataset/
|-- class_1/
|   |-- img_1.png
|   |-- img_2.png
|   |-- ...
|-- class_2/
|   |-- img_1.png
|   |-- img_2.png
|   |-- ...
|-- class_3/
|   |-- img_1.png
|   |-- img_2.png
|   |-- ...
|-- class_4/
|   |-- img_1.png
|   |-- img_2.png
|   |-- ...
```

Put at least one config file into the *config* folder, here an example

```yaml
dataset:
  folder: /all_data/dataset/[DATASET NAME]
  resize_width: 256
  resize_height: 256
  train_size: 0.7
  test_size: 0.15
  valid_size: 0.15
  num_workers: 8
train:
  model_name: resnet18
  base_model_path: /all_data/base_model/
  batch_size: 8 
  epochs: 15
  folder_pth: /all_data/trained_models/
  learning_rate: 0.001
evaluate:
  checkpoint_path: /lightning_logs/version_0/checkpoints/[NAME].ckpt
report:
  folder: /all_data/reports/
```

Run the main file

```python
python main.py [train or test or both]
```

*main.py* script can perform only one 3 types of operations: 

- train 

- test

- both 

for each configuration file present in the configuration folder.

So if I have 5 configuration files and I use the *train* operation the system will do 5 training sessions with the data from the respective config file. 

To show graphs use: 

```shell
tensorboard --logdir=lightning_logs/
```

#### References

[PyTorch Lightning 2.1.3 documentation](https://lightning.ai/docs/pytorch/stable/)

[Timm Library](https://timm.fast.ai/)

[TensorBoard Documentation](https://www.tensorflow.org/tensorboard?hl=en)
