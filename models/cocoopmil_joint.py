import copy
import torch
import torch.nn as nn
import os
import tqdm
import wandb
from models.utils.continual_model import ContinualModel, ContinualModelCoCoopMil
#from utils.attention_maps import test_localization_coopmil
from utils.args import *
#from utils.evaluate import evaluate_val, EarlyStopping
#from utils.evaluate import evaluate_test
from argparse import ArgumentParser
def get_parser():
    parser = ArgumentParser(description='joint')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # parser.add_argument('--dualcoopmultiscale', default=1, choices=[0, 1], type=int, help='Use multiscale')
    parser.add_argument('--passing_v', default=True, type=bool, help='passing v')
    parser.add_argument('--n_ctx', default=32, type=int, help='Use multiscale')
    return parser


class CoCoopMilJoint(ContinualModelCoCoopMil):
    NAME = 'cocoopmil_joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CoCoopMilJoint, self).__init__(backbone, loss, args, transform)
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce= torch.nn.CrossEntropyLoss()

    def observe(self, inputs0, inputs1, labels):
        #self.opt.zero_grad()
        #outputs = self.net([inputs0, inputs1])
        #loss = self.criterion(outputs[0].unsqueeze(0), labels.cuda().view(-1))
        #loss.backward()
        #self.opt.step()
        #if self.args.nowand == 0:
        #    wandb.log({'loss': loss.item()})
        #return loss.item()
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
        cocoop_logits, _, _, prediction_bag, prediction_patch, _ = outputs
        max_prediction, index = torch.max(prediction_patch, 0)
        if not self.args.loss_on_task_logits:
            if self.args.n_classes>1:
                one_hot_labels = torch.nn.functional.one_hot(labels,num_classes=self.args.n_classes)
            else:
                one_hot_labels = labels.unsqueeze(0)
            one_hot_labels = one_hot_labels.float().cuda()

            loss_max = self.bce(max_prediction.view(1, -1), one_hot_labels.view(1, -1))
            loss_bag = self.bce(prediction_bag, one_hot_labels)
            if self.args.loss=="ce":
                loss_prompt = self.ce(cocoop_logits, labels)
            else:
                loss_prompt = self.bce(cocoop_logits, one_hot_labels)
            dsmil_loss = loss_max + loss_bag
            wandb.log({'dualcoop_loss': loss_prompt.item(),'dsmil_loss':dsmil_loss.item()})
            loss = dsmil_loss + loss_prompt
        else:
            #select logits of the currect task
            task_label=labels//self.args.n_classes_per_task
            task_labels = labels%self.args.n_classes_per_task
            max_prediction= max_prediction[task_label*2:task_label*2+2].view(1, -1)
            prediction_bag= prediction_bag[:,task_label*2:task_label*2+2].view(1, -1)
            cocoop_logits= cocoop_logits[:,task_label*2:task_label*2+2].view(1, -1)
            one_hot_labels = torch.nn.functional.one_hot(task_labels,num_classes=self.args.n_classes_per_task)
            one_hot_labels = one_hot_labels.float().cuda()
            loss_max = self.bce(max_prediction.view(1, -1), one_hot_labels.view(1, -1))
            loss_bag = self.bce(prediction_bag, one_hot_labels)
            if self.args.loss=="ce":
                loss_prompt = self.ce(cocoop_logits, task_labels)
            else:
                loss_prompt = self.bce(cocoop_logits, one_hot_labels)
            dsmil_loss = loss_max + loss_bag
            wandb.log({'dualcoop_loss': loss_prompt.item(),'dsmil_loss':dsmil_loss.item()})
            loss = dsmil_loss + loss_prompt

        return loss


    #def end_task(self, dataset, fold):
    #    print("end task", self.NAME)
    #    train_loader,_,test_loader = dataset.get_joint_data_loaders(fold)
    #    early_stopping = EarlyStopping(patience=5, stop_epoch=50, verbose=True,start_epoch=int((3/5)*self.args.n_epochs))
    #    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs, 0.000005)
    #    table= wandb.Table(columns=["metric", "value", "step"])
    #    for epoch in range(self.args.n_epochs):
    #        for idx, data in enumerate(tqdm.tqdm(train_loader)):
    #            if len(data) == 5:
    #                inputs0, inputs1, labels, x_coord, y_coord = data
    #            else:
    #                inputs0, inputs1, labels = data
    #            inputs0 = inputs0.cuda()
    #            inputs1 = inputs1.cuda()
    #            idx = torch.randperm(inputs1.size(1))[:self.args.dropout_patch]
    #            inputs0 = inputs0[:, idx, :]
    #            inputs1 = inputs1[:, idx, :]
    #            labels = labels.cuda()
    #            self.opt.zero_grad()
    #            output = self.net([inputs0, inputs1],table=table, add_context=epoch>self.args.n_epochs//3)
    #            #output = self.net([inputs0, inputs1],table=table, add_context=True)
    #            loss = self.compute_loss(output, labels)
    #            loss.backward()
    #            self.opt.step()
    #            #self.opt2.step()
    #            if self.args.nowand == 0:
    #                wandb.log({'loss': loss.item(),"lr":scheduler.get_last_lr()[0]})
    #        stop = evaluate_val(self, dataset, 0, epoch=epoch, results_dir=wandb.run.dir, early_stopping=early_stopping)
    #        accs = evaluate_test(self, dataset,random=False,logit_to_evaluate=2)
    #        if stop:
    #            break
    #        if scheduler is not None:
    #            scheduler.step()
    #    #early_stopping.load_checkpoint(self, os.path.join(wandb.run.dir, "checkpoint.pt"))
    #    #early_stopping.delete_checkpoint( os.path.join(wandb.run.dir, "checkpoint.pt"))
