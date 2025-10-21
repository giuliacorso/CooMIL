import torch.nn as nn
from backbone.utils.modules import IClassifier, BClassifier, MILNet
import torch
from transformers import CLIPModel
import copy
__all__ = ['MILNetOriginal']


class MILNetOriginal(nn.Module):
    def __init__(self, args, clip_model=None):
        super(MILNetOriginal, self).__init__()
        self.args = args
        self.old_keys, self.old_dsmil = [], []
        if self.args.dsmil_multiscale:
            scale = 2
        else:
            scale = 1
        self.keys = nn.Parameter(nn.init.normal_(torch.empty(4, 512), std=0.02))
        self.dsmil = MILNet(IClassifier(args.input_size*scale, args.n_classes), BClassifier(args.input_size*scale,  args.n_classes), args)

    def combine_old_new_predictions(self,old_bag_predictions,new_bag_predictions):
        old_bag_predictions = torch.cat([item[:,idx*2:(idx+1)*2] for idx, item in enumerate(old_bag_predictions)], dim=1)
        new_bag_predictions = new_bag_predictions[:,len(self.old_dsmil) * 2:]
        bag_predictions = torch.cat([old_bag_predictions, new_bag_predictions], dim=1)
        return bag_predictions

    def forward(self, x):
        if len(self.old_dsmil) > 0:
            old_bag_predictions = []
            old_patches_predictions = []
            for idx, old_dsmil in enumerate(self.old_dsmil):
                old_prediction_bag, y_prob, _, _, classes, _,_ = old_dsmil(x)
                old_bag_predictions.append(old_prediction_bag)
                old_patches_predictions.append(classes)

            new_prediction_bag, y_prob, y_hat, A, new_prediction_patch, B, proj= self.dsmil(x)
            bag_predictions = self.combine_old_new_predictions(old_bag_predictions, new_prediction_bag)
            patches_predictions = self.combine_old_new_predictions(old_patches_predictions, new_prediction_patch)
        else:
            bag_predictions, y_prob, y_hat, A, patches_predictions, B, proj= self.dsmil(x)
        #proj=proj@ self.visual_projection.T
        #proj = proj / proj.norm(dim=-1, keepdim=True)
        keys= self.get_keys()
        return  y_prob, y_hat,bag_predictions, A, patches_predictions, B, keys,proj

    def freeze_keys(self, task):
        self.old_keys.append(self.keys[task].detach().clone())
        print("Freezing keys", str(task))
        if not self.args.single_dsmil:
            old_dsmil = copy.deepcopy(self.dsmil)
            for param in old_dsmil.parameters():
                param.requires_grad = False
            self.old_dsmil.append(old_dsmil)

    def get_keys(self):
        if len(self.old_keys) > 0:
            new = self.keys[len(self.old_keys):]
            keys = torch.cat([torch.stack(self.old_keys), new])
        else:
            keys = self.keys
        return keys

