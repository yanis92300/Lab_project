# Lab-project : using Segment Anything (SAM) to segment cancer cells in H&E Images
This repository contains the code implementation of our lab project which consists of investigating how [SAM](https://github.com/facebookresearch/segment-anything.git) can be leveraged for automatic cancer cell detection in H&E images. 

Specifically, our contribution is two-fold:
1. We show how te performances of SOTA model [Hovernet](https://github.com/vqdang/hover_net.git) can be improved with SAM used as a post-processing 
2. We combine the efforts made in the works of [MedSAM](https://github.com/bowang-lab/MedSAM.git) and [CellViT](https://github.com/TIO-IKIM/CellViT.git) to slightly improve the SOTA performances in automated instance segmentation of cell nuclei in digitized tissue samples. More specifically, we use the weights of the ViT encoder from MedSAM in the training of CellViT.

Both experiments are made on the [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) dataset, a challenging nuclei instance segmentation benchmark.

## Installation

Clone the repository: ``` git clone https://github.com/olavdc/Lab-project.git ```

Pip install the requirements : ``` pip install -r requirements. txt ```

## Usage

### Project structure
```
├── base_ml               # Basic Machine Learning Code: CLI, Trainer, Experiment, ...
├── cell_segmentation     # Cell Segmentation training and inference files
│   ├── datasets          # Datasets (PyTorch)
│   ├── experiments       # Specific Experiment Code for different experiments
│   ├── inference         # Inference code for experiment statistics and plots
│   ├── trainer           # Trainer functions to train networks
│   ├── utils             # Utils code
│   └── run_xxx.py        # Run file to start an experiment
├── configs               # Config files
│   ├── examples          # Example config files with explanations
│   └── python            # Python configuration file for global Python settings
├── datamodel             # Datamodels of WSI, Patientes etc. (not ML specific)
├── docs                  # Documentation files (in addition to this main README.md)
├── models                # Machine Learning Models (PyTorch implementations)
│   ├── encoders          # Encoder networks (see ML structure below)
│   ├── pretrained        # Checkpoint of important pretrained models (needs to be downloaded from Google drive)
│   └── segmentation      # CellViT Code
├── preprocessing         # Preprocessing code
│   └── patch_extraction  # Code to extract patches from WSI
```
### Pannuke dataset prepration

To preprocess the Pannuke dataset in order to have the right input for the model, it is necessary to convert the Pannuke dataset which is originally in the following format:

```
├── fold0
│   ├── images.npy
│   ├── masks.npy
│   └── types.npy
├── fold1
│   ├── images.npy
│   ├── masks.npy
│   └── types.npy
└── fold2
    ├── images.npy
    ├── masks.npy
    └── types.npy
```
into a dataset in the following format which is more suitable for multithreading and the application of data augmentation:

```
├── fold0
│   ├── cell_count.csv      # cell-count for each image to be used in sampling
│   ├── images              # H&E Image for each sample as .png files
│   ├── images
│   │   ├── 0_0.png
│   │   ├── 0_1.png
│   │   ├── 0_2.png
...
│   ├── labels              # label as .npy arrays for each sample
│   │   ├── 0_0.npy
│   │   ├── 0_1.npy
│   │   ├── 0_2.npy
...
│   └── types.csv           # csv file with type for each image
├── fold1
│   ├── cell_count.csv
│   ├── images
│   │   ├── 1_0.png
...
│   ├── labels
│   │   ├── 1_0.npy
...
│   └── types.csv
├── fold2
│   ├── cell_count.csv
│   ├── images
│   │   ├── 2_0.png
...  
│   ├── labels  
│   │   ├── 2_0.npy
...  
│   └── types.csv  
├── dataset_config.yaml     # dataset config with dataset information
└── weight_config.yaml      # config file for our sampling
```

In order to convert the data into the correct format, we invite you to run the following commands from the directory (~/CellViT/cell_segmentation/dataset):: 

```
python prepare_pannuke.py --input_path INPUT_PATHv--output_path OUPUT_PATH

required named arguments:
--input_path INPUT_PATH original Pannuke dataset path
--output_path OUPUT_PATH processed Pannuke dataset path
```
### Training

#### Training from scratch

To replicate the results presented in our report from scratch, you will first need to download the weights of the classical SAM encoder wieghts (i.e. SAM-ViT-B) and the MedSAM encoder weights (i.e. MedSAM-ViT-B) , which you can find at this [link](https://drive.google.com/drive/folders/1HKZUDm1SZejdVYZKlbb8ufsACjfx8Pcd?usp=drive_link)

Defining which encoder weights (i.e., SAM-ViT-B or MedSAM-ViT-B) to use during the training of CellVit will be done by writing a config.yaml file, in which you will need to specify the desired encoder weights for training. The configuration file should have the following format, and it is under the "pretrained_encoder" subsection of the "model" section where you should define which encoder weights to use:

```
logging:
  mode: online
  project: Cell-Segmentation
  notes: CellViT-SAM-H
  log_dir: ~/CellViT/cell_segmentation/experiments/ # Directory where you want the logs to be saved
  log_comment: CellViT-SAM-B
  tags:
  - Fold-1
  - SAM-H
  wandb_dir: ./CellViT/wandb_results/
  level: Debug
  group: CellViT-SAM-H
random_seed: 19
gpu: 0
data:
  dataset: PanNuke
  dataset_path: ~/CellViT/configs/datasets/PanNuke # Direcotry where we can find the preprocessed Pannuke dataset
  train_folds:
  - 0
  val_folds:
  - 1
  test_folds:
  - 2
  num_nuclei_classes: 6
  num_tissue_classes: 19
model:
  backbone: SAM-B
  pretrained_encoder: ~/CellViT/SAM_ViT_B.pth # Here we precise which encoder weights we want to use:
                                                                            (i) if you want the classical SAM encoder weights: use SAM_ViT_B.pth
                                                                            (ii) if you want to use the MedSAM encoder wiehts: use MEDSAM_ViT_B.pth
  shared_skip_connections: true
loss:
  nuclei_binary_map:
    focaltverskyloss:
      loss_fn: FocalTverskyLoss
      weight: 1
    dice:
      loss_fn: dice_loss
      weight: 1
  hv_map:
    mse:
      loss_fn: mse_loss_maps
      weight: 2.5
    msge:
      loss_fn: msge_loss_maps
      weight: 8
  nuclei_type_map:
    bce:
      loss_fn: xentropy_loss
      weight: 0.5
    dice:
      loss_fn: dice_loss
      weight: 0.2
    mcfocaltverskyloss:
      loss_fn: MCFocalTverskyLoss
      weight: 0.5
      args:
        num_classes: 6
  tissue_types:
    ce:
      loss_fn: CrossEntropyLoss
      weight: 0.1
training:
  drop_rate: 0
  attn_drop_rate: 0.1
  drop_path_rate: 0.1
  batch_size: 8
  epochs: 40 # To change if you have the necessary hardware 
  optimizer: AdamW
  early_stopping_patience: 130
  scheduler:
    scheduler_type: exponential
    hyperparameters:
      gamma: 0.85
  optimizer_hyperparameter:
    betas:
    - 0.85
    - 0.95
    lr: 0.0003
    weight_decay: 0.0001
  unfreeze_epoch: 25
  sampling_gamma: 0.85
  sampling_strategy: cell+tissue
  mixed_precision: true
transformations:
  randomrotate90:
    p: 0.5
  horizontalflip:
    p: 0.5
  verticalflip:
    p: 0.5
  downscale:
    p: 0.15
    scale: 0.5
  blur:
    p: 0.2
    blur_limit: 10
  gaussnoise:
    p: 0.25
    var_limit: 50
  colorjitter:
    p: 0.2
    scale_setting: 0.25
    scale_color: 0.1
  superpixels:
    p: 0.1
  zoomblur:
    p: 0.1
  randomsizedcrop:
    p: 0.1
  elastictransform:
    p: 0.2
  normalize:
    mean:
    - 0.5
    - 0.5
    - 0.5
    std:
    - 0.5
    - 0.5
    - 0.5
eval_checkpoint : best_checkpoint
run_sweep: false
agent: null
dataset_config:
  tissue_types:
    Adrenal_gland: 0
    Bile-duct: 1
    Bladder: 2
    Breast: 3
    Cervix: 4
    Colon: 5
    Esophagus: 6
    HeadNeck: 7
    Kidney: 8
    Liver: 9
    Lung: 10
    Ovarian: 11
    Pancreatic: 12
    Prostate: 13
    Skin: 14
    Stomach: 15
    Testis: 16
    Thyroid: 17
    Uterus: 18
  nuclei_types:
    Background: 0
    Neoplastic: 1
    Inflammatory: 2
    Connective: 3
    Dead: 4
    Epithelial: 5
```
Then, once you are in the directory (~/CellViT/cell_segmentation), execute the following command line:
```
python run_cellvit.py --config GONIF [--gpu GPU] 

optional arguments:
  --gpu GPU    Cuda-GPU ID (default: None)
required named arguments:
  --config CONFIG    Path to a config file (default: None)
```
#### Training from a checkpoint

To continue training from a previously trained checkpoint, it will first be necessary to download the training checkpoint according to the experiment you are conducting. In the case of experimenting with the classic SAM encoder, download the weight checkpoint_epoch_40_SAM_ViT_B.pth, and if you are experimenting with the MedSAM encoder, download the weight checkpoint_epoch_40_MEDSAM_ViT_B.pth from this [link](https://drive.google.com/drive/folders/1PfB0x-tqec5cAI74xydi3znBO0Eve1TI?usp=sharing). Make sure you specify the appropriate encoder weights in your config.yaml file and execute the following command from the directory (~/CellViT/cell_segmentation):

```
python run_cellvit.py --config GONIF [--gpu GPU] --checkpoint CHECKPOINT

optional arguments:
--gpu GPU    Cuda-GPU ID (default: None)
--checkpoint CHECKPOINT    Path to where tou stored the checkpoints
required named arguments:
--config CONFIG    Path to a config file (default: None)
```

