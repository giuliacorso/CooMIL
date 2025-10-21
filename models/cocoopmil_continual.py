import copy
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from utils.training import evaluate
from models.utils.continual_model import ContinualModel, ContinualModelCoCoopMil
from utils.args import *
#from utils.attention_maps import test_localization_coopmil
#from utils.attention_maps import test_localization_coopmil_cam
from sklearn.cluster import KMeans
from argparse import ArgumentParser
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning coMIl')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # parser.add_argument('--dualcoopmultiscale', default=1, choices=[0, 1], type=int, help='Use multiscale')
    parser.add_argument('--passing_v', default=True, type=bool, help='passing v')
    parser.add_argument('--n_ctx', default=32, type=int, help='Use multiscale')

    return parser


class CoCoopMilContinual(ContinualModelCoCoopMil):
    NAME = 'cocoopmil_continual'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CoCoopMilContinual, self).__init__(backbone, loss, args, transform)
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce= torch.nn.CrossEntropyLoss()
        self.current_task=0

    def observe(self, inputs0, inputs1, labels):
        self.opt.zero_grad()
        idx = torch.randperm(inputs1.size(1))[:self.args.dropout_patch]
        inputs0 = inputs0[:, idx, :]
        inputs1 = inputs1[:, idx, :]
        labels = labels.cuda()
        self.opt.zero_grad()
        output = self.net([inputs0, inputs1])
        loss = self.compute_loss(output, labels)
        loss.backward()
        self.opt.step()
        return loss.item()

    def compute_loss(self, outputs, labels):
        tot_classes = (self.current_task+1)*2
        cocoop_logits, _, _, prediction_bag, prediction_patch, _ = outputs
        prediction_bag = prediction_bag[:,0:tot_classes]
        prediction_patch = prediction_patch[:,0:tot_classes]
        cocoop_logits = cocoop_logits[:,0:tot_classes]
        max_prediction, index = torch.max(prediction_patch, 0)
        one_hot_labels = torch.nn.functional.one_hot(labels.squeeze().unsqueeze(0),
                                                     num_classes=tot_classes).float().cuda()
        loss_max = self.bce(max_prediction.view(1, -1), one_hot_labels.view(1, -1))
        loss_bag = self.bce(prediction_bag, one_hot_labels)
        loss_prompt = self.ce(cocoop_logits, labels)
        dsmil_loss = loss_max + loss_bag
        wandb.log({'dualcoop_loss': loss_prompt.item(), 'dsmil_loss': dsmil_loss.item()})
        loss = dsmil_loss + loss_prompt
        return loss

    def begin_task(self,train_loader):
        #train_loader = dataset.train_loaders[self.current_task]
        datas=[]
        for data in train_loader:
            inputs0, inputs1, labels = data
            datas.append(inputs0.view(-1,512).mean(0).unsqueeze(0).cpu().numpy())
        X=np.concatenate(datas)
        kmeans = KMeans(n_clusters=self.args.n_clusters, random_state=0, n_init="auto").fit(X)
        clusters=kmeans.cluster_centers_
        self.net.clusters[self.current_task*self.args.n_clusters:(self.current_task+1)*self.args.n_clusters]=torch.Tensor(clusters)



    def end_task(self, dataset):
        print("end task", self.NAME)
        self.net.prompt_learner.freeze_prompts(self.current_task, 2)
        self.net.freeze_keys(self.current_task)
        self.current_task += 1


    def check_sums(self, list_old_parameters, name):
        for idx, model in enumerate(list_old_parameters):
            sum = 0
            if isinstance(model,torch.Tensor):
                sum = model.sum()
            else:
                for param in model.parameters():
                    sum = sum + param.sum()
            wandb.log({"sum_old_" + name + str(idx): sum})
        print("check sum"+ name)


