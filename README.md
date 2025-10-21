# CooMIL
Context-guided Prompt-learning for Continual WSI Classification

## Data Preparation

The [CLAM](https://github.com/mahmoodlab/CLAM) pipeline is followed, in particular the following code is used to extract tiles of regions and patches:
```
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --seg --patch --stitch 
```

The patch size in the final output is determined by the combination of parameters:  
- `--patch_size`: defines the size of the extracted patch at the selected resolution level.  
- `--step_size`: controls the stride between consecutive patches (set equal to `--patch_size` for non-overlapping patches).  

Regions and patches are extracted at the highest available resolution (**40x, segmentation level 0**). Each region corresponds to a **4096 × 4096** area of the WSI (with `--patch_size` and `--step_size` set to 4096), while the extracted patches to **512 × 512** pixels (with `--patch_size` and `--step_size` set to 512).

Once the tiles are extracted, the user needs to run the script `preprocessing/feature_extraction.py` to compute the features for both regions and patches.  

Starting from these features, the final preprocessing step is to associate the features of each region with the features of the patches contained within it, preserving their spatial relationships. The result are saved as one pickle file for each WSI. To perform this step, the user needs to run the script `preprocessing/feature_aggregation.py`.

The repository contains one CSV file for each tissue (breast, esophagus, kidney, lung), named as `tissue10fold_conch.csv`, structured as follows:
```
slide,labels,fold
```
Each CSV file is designed to serve as input for the following phase. Make sure to generate these input files so that they include:  

- the path to the pickle file of each WSI, which contains the multi-resolution embeddings extracted with [CONCH](https://github.com/mahmoodlab/CONCH.git)  
- the label of the WSI, which is a number from 0 to 7
- the test fold associated with each WSI, which is a number from 0 to 9

Finally, the user needs to customize the `Sequential_Generic_MIL_Dataset` class in  
`CooMIL/datasets/seq_wsi.py` based on the tumor types and the number of tasks to test.  

Specifically, the following attributes must be updated:  
- `self.N_TASKS`: number of tumor types considered  
- `self.datasets`: list of datasets, with the corresponding `name` and `csv_path` for each tumor  
- `self.class_names`: class labels for all tumor subtypes included  
- `self.task_names`: names of the tumor-level tasks  

For example, the default implementation is set for the **four tumor types explored in the study**:

```python
self.N_TASKS = 4
self.datasets = [
    Generic_MIL_Dataset(name="lung", csv_path='lung10fold_conch.csv',  args=self.args),
    Generic_MIL_Dataset(name="brca", csv_path='brca10fold_conch.csv', args=self.args),
    Generic_MIL_Dataset(name="kidney", csv_path='kidney10fold_conch.csv', args=self.args),
    Generic_MIL_Dataset(name="esca", csv_path='esca10fold_conch.csv', args=self.args)
]
self.class_names = ["Lung Adenocarcinoma", "Lung squamous cell carcinoma", 
                    "Breast Invasive ductal","Breast Invasive lobular", 
                    "Kidney clear cell carcinoma", "Kidney papillary cell carcinoma",
                    "Esophageal adenocarcinoma", "Esophageal squamous cell carcinoma"]
self.task_names = ["Lung", "Breast", "Kidney", "Esca"]
```
If you plan to experiment with the same tumor types, you only need to update the csv_path values to match the generated input files.
If you want to test a different set of tumors, make sure to adapt all the attributes (self.N_TASKS, self.datasets, self.class_names, self.task_names) accordingly.

## Training Example

The following commands show how to launch the training for each model included in the framework, both for the proposed method and for the competitor baselines.


```
python -u utils/main.py --model cocoopmil_continual --dataset seq-wsi --exp_desc cocoopmil_continual --n_epochs 50
python -u utils/main.py --model cocoopmil_naive --dataset seq-wsi --exp_desc cocoopmil_naive --n_epochs 50
python -u utils/main.py --model cocoopmil_joint --dataset seq-wsi --exp_desc cocoopmil_joint --n_epochs 200

python -u utils/main.py --model gdumb --dataset seq-wsi --exp_desc gdumb --buffer_size 1100 
python -u utils/main.py --model er_ace --dataset seq-wsi --exp_desc er_ace --buffer_size 1100 --n_epochs 50 

python -u utils/main.py --model lwf --dataset seq-wsi --exp_desc lwf --alpha 0.2
python -u utils/main.py --model ewc_on --dataset seq-wsi --exp_desc ewc_on --e_lambda 0.1 --gamma 0.1
python -u utils/main.py --model derpp --dataset seq-wsi --exp_desc derpp --alpha 0.2 --beta 0.2 --n_epochs 50 --buffer_size 1100
python -u utils/main.py --model derpp --dataset seq-wsi --exp_desc derpp --alpha 0.2 --beta 0.2 --n_epochs 50 --buffer_size 0

python -u utils/main.py --model conslide --dataset seq-wsi --exp_desc conslide --alpha 0.2 --beta 0.2 --n_epochs 50 --buffer_size 1100
python -u utils/main.py --model conslide --dataset seq-wsi --exp_desc conslide --alpha 0.2 --beta 0.2 --n_epochs 50 --buffer_size 0 

```

## Acknowledgements

Framework code for Continual Learning was largely adapted via making modifications to [ConSlide](https://github.com/HKU-MedAI/ConSlide)
