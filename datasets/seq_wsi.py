from __future__ import print_function, division
from asyncio import base_tasks
import os
import torch
import numpy as np
import joblib
import glob 
import tqdm
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
import collections
from itertools import islice
import bisect
import torch.nn.functional as f
from torch.utils.data import Dataset
import h5py

import torch.nn.functional as F
from torch.utils.data import DataLoader
# from cl_wsi import datasets
from utils.conf import base_path
# from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
#from backbone.model_clam import CLAM_SB
from backbone.hit import HIT
from backbone.cocoopmil import CoCoopMil 
#from backbone.transmil import TransMIL
#from backbone.dsmil import FCLayer, BClassifier, MILNet
import random
from sklearn.model_selection import train_test_split
from utils.conf import seed_worker, set_random_seed

#def seed_worker(worker_id):
#    # Generate a seed for the worker based on the initial seed
#    worker_seed = torch.initial_seed() % 2**32
#    # Set the seed for NumPy operations in the worker
#    np.random.seed(worker_seed)
#    # Set the seed for random number generation in the worker
#    random.seed(worker_seed)
    
def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    img2 = torch.cat([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2] for item in batch])
    if len(batch[0]) > 3:
        logits = torch.cat([item[3] for item in batch], dim = 0)
        return [img, img2, label, logits]
    return [img, img2, label]

# from utils.utils import generate_split, nth

def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
    seed = 7, label_frac = 1.0, custom_test_ids = None):
    indices = np.arange(samples).astype(int)
    
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            
            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids

def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

    df.to_csv(filename)
    print()
class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self, csv_path='/homes/gbontempo/continual-MIL/data/mil', name="Elephant", args=None):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            name: name of the dataset
            args: arguments
        """
        self.num_classes = 2
        self.seed = args.seed
        self.name = name
        self.args = args
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if len(csv_path) > 0:
            self.slide_data = pd.read_csv(csv_path)
 

    def __len__(self):
        return self.slide_data.shape[0]
    
    def reset_label(self,data):
        if self.args.cam=="reverse_order":
            data.labels= abs(data.labels-7)
        if self.args.cam=="brca":
            data.labels= data.labels-2
        return data

    def return_splits(self):
        data = pd.DataFrame(self.slide_data)
        data=self.reset_label(data)
        if self.args.debug_mode:
            train_ids, val_ids, test_ids = self.generate_split_debug(data)
        else:
            ids = list(range(10))
            ids.remove(self.args.test_fold)
            ids.remove(self.args.val_fold)
            self.val_fold=self.args.val_fold
            print('Val fold id:',self.args.val_fold)
            print('Test fold id:',self.args.test_fold)
            
            train_ids, val_ids, test_ids = self.generate_split(data)
        assert len(set(train_ids).intersection(val_ids)) == 0
        assert len(set(val_ids).intersection(test_ids)) == 0

        train_dataset = Generic_Split(data.iloc[train_ids], data_dir=self.data_dir, args=self.args)
        val_dataset = Generic_Split(data.iloc[val_ids], data_dir=self.data_dir, args=self.args)
        test_dataset = Generic_Split(data.iloc[test_ids], data_dir=self.data_dir, args=self.args)
        return train_dataset, val_dataset, test_dataset

    def __getitem__(self, idx):
        return None

    def generate_split(self, data):

        # test id if fold is one of the first three folds
        #test_filter = data["fold"] < 3
        #val_filter = data["fold"] == fold
        #train_filter = (data["fold"] > 3) & (data["fold"] != fold)

        
        test_filter = data["fold"] == self.args.test_fold
        val_filter = data["fold"] == self.val_fold
        train_filter = (data["fold"] != self.args.test_fold) & (data["fold"] != self.val_fold)

        test_ids = data[test_filter].index
        val_ids = data[val_filter].index
        train_ids = data[train_filter].index
        return train_ids, val_ids, test_ids


    def split_indices(self, df):
        train_idx, temp_idx = train_test_split(df.index, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        return train_idx, val_idx, test_idx

    def generate_split_debug(self, data):
        vals = np.unique(data['labels'].values)
        class_A = data[self.slide_data['labels'] == vals[0]][:30]
        class_B = data[self.slide_data['labels'] == vals[1]][:30]
        train_ids_A, val_ids_A, test_ids_A = self.split_indices(class_A)
        train_ids_B, val_ids_B, test_ids_B = self.split_indices(class_B)

        train_ids = list(train_ids_A) + list(train_ids_B)
        val_ids = list(val_ids_A) + list(val_ids_B)
        test_ids = list(test_ids_A) + list(test_ids_B)

        return train_ids, val_ids, test_ids
        

class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
                 csv_path,
                 **kwargs):
        super(Generic_MIL_Dataset, self).__init__(csv_path=csv_path, **kwargs)
        self.data_dir = csv_path
        self.slides = []

    def __getitem__(self, idx):
        #label = self.slide_data.iloc[idx][1]
        row = self.slide_data.iloc[idx]
        label = row.iloc[1]
        if self.args.loadonmemory:
            slide = self.slides[idx]
            patch_embeddings, region_embeddings, label = slide[0], slide[1], label
        else:
            #features = self.slide_data.iloc[idx][0].replace(" ", "")
            features = row.iloc[0]

            #data = joblib.load(features[11:])
            data = joblib.load(features)
            patch_embeddings, region_embeddings, label = data["patch"].numpy(), data["region"].numpy(), label

        return torch.Tensor(patch_embeddings), torch.Tensor(region_embeddings), label


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, args=None):
        super(Generic_Split, self).__init__(csv_path="", args=args)
        self.args = args
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.slides = []

        if args.loadonmemory:
            for idx, slide in tqdm.tqdm(enumerate(self.slide_data["slide"].values)):
                #slide = slide.replace(" ", "")
                #slide = slide[11:]
                data = joblib.load(slide)
                self.slides.append((data["patch"].numpy(), data["region"].numpy()))

    def __len__(self):
        return len(self.slide_data)


# class Generic_Split(Generic_MIL_Dataset):
#     def __init__(self, slide_data, data_dir=None, num_classes=2):
#         self.use_h5 = False
#         self.slide_data = slide_data
#         self.data_dir = data_dir
#         self.num_classes = num_classes
#         self.slide_cls_ids = [[] for i in range(self.num_classes)]
#         for i in range(self.num_classes):
#             self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

#     def __len__(self):
#         return len(self.slide_data)

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    # @property
    # def cummulative_sizes(self):
    #     warnings.warn("cummulative_sizes attribute is renamed to "
    #                   "cumulative_sizes", DeprecationWarning, stacklevel=2)
    #     return self.cumulative_sizes
        
def process_slide(slide_path):
    if not os.path.isfile(os.path.join(slide_path, "sortedv3.joblib")):
        patches = joblib.load(os.path.join(slide_path, "embeddings.joblib"))
        patch_level = patches["level"].to_numpy()
        patch_childof = patches["childof"].to_numpy()
        patch_childof[np.isnan(patch_childof)] = -1
        embeddings = patches["embedding"]
        x_coords = torch.LongTensor(patches["x"])
        y_coords=torch.LongTensor(patches["y"])
        x_coords=x_coords[patch_level==3]
        y_coords=y_coords[patch_level==3]
        size = embeddings.shape[0]
        # Get X
        x = []
        for i in range(size):
            x.append(torch.Tensor(np.matrix(embeddings[i])))
        X = torch.vstack(x)

        # Save label
        label = os.path.basename(slide_path).split("_")[-1]
        if "0" in label or "1" in label:
            label = int(label)
        else:
            if label == "tumor":
                label = 1
            else:
                label = 0
        indecesperlevel = []
        # forward input for each scale gnn
        for i in np.unique(patch_level):
            # select scale
            indeces_feats = torch.Tensor((patch_level == i).nonzero()[0]).int().view(-1)
            indecesperlevel.append(indeces_feats)
        child_index = indecesperlevel[1]
        parents_index = patch_childof[child_index]

        featshigher = X[child_index]
        featslower = X[parents_index]
        print(featshigher.size(), featslower.size())
        joblib.dump((featshigher, featslower, label,x_coords,y_coords), os.path.join(slide_path, "sortedv3.joblib"))
    else:
        featshigher, featslower, label, x_coords, y_coords= joblib.load(os.path.join(slide_path, "sortedv3.joblib"))
    return featshigher, featslower, label,x_coords, y_coords

class Split(Dataset):
    def __init__(self, data,args,name="train"):
        super(Split, self).__init__()
        self.name = name
        self.paths = data
        self.args=args
        self.slide_data=pd.DataFrame([int(os.path.basename(slide)[-1]) for slide in self.paths],columns=["labels"])
        self.data=[]
        if self.args.loadonmemory:
            for slide in self.paths:
                self.data.append(process_slide(slide))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.args.loadonmemory:
            return self.data[idx]
        else:
            return process_slide(self.paths[idx])

class CamDataset(Dataset):
    """
    Custom dataset for processing and accessing data.

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): A function/transform that takes in an
            object and returns a transformed version. Defaults to None.
        pre_transform (callable, optional): A function/transform that takes in an
            object and returns a transformed version. Defaults to None.
        type (str, optional): Type of the dataset. Defaults to "train".
    """

    def __init__(self,source,args):
        super(CamDataset, self).__init__()
        self.source = source
        self.bags = glob.glob(os.path.join(source, "*/*"))
        self.args = args
        self.process()

    def len(self):
        return len(self.bags)

    def process(self):
        """
        Process the dataset and save processed data.
        """
        bags = glob.glob(os.path.join(self.source, "*/*"))
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        for idx, bag in enumerate(bags):
            if "test" in bag:
                if self.args.debug_mode and len(self.test_dataset)>10:
                    continue
                self.test_dataset.append(bag)
            elif "val" in bag:
                if self.args.debug_mode and len(self.val_dataset) > 10:
                    continue
                self.val_dataset.append(bag)
            else:
                if self.args.debug_mode and len(self.train_dataset) > 10:
                    continue
                self.train_dataset.append(bag)

    def __getitem__(self, idx):
        pass

    def return_splits(self):
        """
        Return the dataset split.

        Args:
            fold (int): Fold number.

        Returns:
            tuple: Train, validation and test datasets.
        """
        train_size = int(len(self.train_dataset) * 0.98)
        val_size = len(self.train_dataset) - train_size
        self.val_dataset = self.train_dataset[:val_size]
        self.train_dataset = self.train_dataset[val_size:]
        return Split(self.train_dataset,self.args,"train"), Split(self.val_dataset,self.args,"val"), Split(self.test_dataset,self.args,"test")




def print_summary(data, name):
    print("---------------------------------")
    print(f"{name} dataset summary")
    print(f"Number of slides: {len(data)}")
    classes = data["labels"].value_counts()
    name_classes = data["labels"].unique()
    print(f"Classes: {name_classes}")
    for name in name_classes:
        print(f"Number of {name} slides: {classes[name]}")

    print("---------------------------------")


class Sequential_Generic_MIL_Dataset(ContinualDataset):
    # NAME = 'seq-milv2'
    NAME = "seq-wsi"
    SETTING = 'class-il'
    # SETTING = 'task-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 4
    TRANSFORM = None

    # FOLD = 0

    def __init__(self, args):

        super(Sequential_Generic_MIL_Dataset, self).__init__(args)
        if args.cam=="normal_order":
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
        elif args.cam=="reverse_order":
            self.N_TASKS = 4
            self.datasets = [
                Generic_MIL_Dataset(name="esca", csv_path='esca10fold_conch.csv', args=self.args),
                Generic_MIL_Dataset(name="kidney", csv_path='kidney10fold_conch.csv', args=self.args),
                Generic_MIL_Dataset(name="brca", csv_path='brca10fold_conch.csv', args=self.args),
                Generic_MIL_Dataset(name="lung", csv_path='lung10fold_conch.csv',  args=self.args)
            ]
            self.class_names = [ "Esophageal squamous cell carcinoma","Esophageal adenocarcinoma",
                                "Kidney papillary cell carcinoma","Kidney clear cell carcinoma",
                                "Breast Invasive lobular", "Breast Invasive ductal",
                                "Lung squamous cell carcinoma", "Lung Adenocarcinoma",
                                
                          ]
            self.task_names = ["Esca","Kidney", "Breast", "Lung"]
        elif args.cam=="lung":
            self.N_TASKS = 1
            self.datasets = [
                Generic_MIL_Dataset(name="lung", csv_path='lung10fold_conch.csv',  args=self.args)
            ]
            self.class_names = ["Lung Adenocarcinoma", "Lung squamous cell carcinoma"]
            self.task_names = [ "Lung"]
        elif args.cam=="brca":
            self.N_TASKS = 1
            self.datasets = [
                Generic_MIL_Dataset(name="brca", csv_path='brca10fold_conch.csv', args=self.args)
            ]
            self.class_names = ["Breast Invasive ductal","Breast Invasive lobular"]
            self.task_names = [ "Breast"]

    def load(self):
        print("Loading data")
        self.test_loaders = []
        self.train_loaders = []
        self.val_loaders = []
        if "joint" in self.args.model:
            self.train_datasets= []
        for n in range(len(self.datasets)):
            dataset = self.datasets[n]
            train_dataset, val_dataset, test_dataset = dataset.return_splits()
            print("---------------------------------")
            print_summary(train_dataset.slide_data, "train")
            print_summary(val_dataset.slide_data, "val")
            print_summary(test_dataset.slide_data, "test")
            print("---------------------------------")
            g = torch.Generator()
            g.manual_seed(self.args.seed)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g)
            if "joint" in self.args.model:
                self.train_datasets.append(train_dataset)
            else:
                self.train_loaders.append(train_loader)
            if self.args.test_on_val:
                self.test_loaders.append(val_loader)
                self.val_loaders.append(test_loader)
            else:
                self.test_loaders.append(test_loader)
                self.val_loaders.append(val_loader)
        if "joint" in self.args.model:
            train_dataset_tot = ConcatDataset(self.train_datasets)
            train_loader = DataLoader(train_dataset_tot, batch_size=1, shuffle=True)
            self.train_loaders.append(train_loader)
        print("Data loaded")


    def get_data_loaders(self, fold):
        train_loader = self.train_loaders[self.i]
        val_loader = self.val_loaders[self.i ]
        test_loader = self.test_loaders[self.i]
        self.i =(self.i+1)%(self.N_TASKS+1)
        return train_loader, val_loader, test_loader

    def get_joint_data_loaders(self, fold):
        train_loader = self.train_loaders[0]
        self.i = self.N_TASKS
        val_loader = self.val_loaders[0]
        test_loader = self.test_loaders[0]
        return train_loader, val_loader, test_loader

    def get_backbone(self, args):
        if 'cocoopmil' in args.model:
            return CoCoopMil(classnames=self.class_names,task_names=self.task_names,args=args)
        else:
            return HIT(num_classes=8)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return f.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None


