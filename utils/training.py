# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import wandb
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False, start_epoch=25):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch = 0
        self.start_epoch = start_epoch

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        start_epoch = self.start_epoch
        score = -val_loss
        if epoch >= start_epoch - 1:
            if self.best_score is None:
                self.best_epoch = epoch
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
            elif score < self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience or epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_epoch = epoch
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
                self.counter = 0
        
        wandb.log({"Epoch": epoch})
    
    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
        self.ckpt_name = ckpt_name

    def load_checkpoint(self, model, ckpt_name):
        model.load_state_dict(torch.load(ckpt_name))
        print(f"Loaded {ckpt_name}")

    def delete_checkpoint(self,path):
        ckpt_name= path
        os.remove(ckpt_name)
        print(f"Deleted {ckpt_name}")


def generate_task_logits(model, proj, current_task, args,
                         num_total_tasks=4, num_classes_per_task=2):

    if current_task > 0:
        clusters = model.net.clusters[:current_task * args.n_clusters].cuda()
        task_labels = torch.cat([i*torch.ones(args.n_clusters, dtype=torch.long) for i in range(current_task)], dim=0)
        query = proj.squeeze().mean(0, keepdim=True)
        mse_dist = (query - clusters).pow(2).sum(1)
        top1_idx = torch.argmin(mse_dist)
        task_id = task_labels[top1_idx]
    else:
        task_id = 0
        mse_dist = torch.zeros(num_total_tasks * num_classes_per_task).cuda()

    mask = [torch.ones(num_classes_per_task) if i == task_id
            else float('-inf') * torch.ones(num_classes_per_task) for i in range(num_total_tasks)]
    mask = torch.cat(mask, dim=0).cuda()

    task_logits = torch.eye(num_total_tasks)[task_id].cuda().view(1, -1)

    return mask, task_logits



def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    # prob_list, labels_list = [], []
    accs, accs_mask_classes = [], []
    aucs, aucs_mask_classes = [], []
    all_prob_list, all_labels_list = [], []
    eye_array = np.eye(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS)
    num_task = len(dataset.test_loaders)

    corrects = 0
    corrects_mask_classes = 0
    totals = 0

    for k, test_loader in enumerate(dataset.test_loaders):
        if k > dataset.i - 1:
            continue

        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        prob_list, mask_prob_list, labels_list = [], [], []
        for data in test_loader:
            
            with torch.no_grad():
                # inputs, labels = data
                # inputs, labels = inputs.to(model.device), labels.to(model.device)
                inputs0, inputs1, labels = data
                inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs0, k)
                else:
                    # import ipdb;ipdb.set_trace()
                    outputs = model([inputs0, inputs1])

                logits, Y_prob, pred = outputs[:3]

                if model.NAME == 'cocoopmil_continual':
                    proj= outputs[-1]
                    current_task = model.current_task
                    _, task_logits = generate_task_logits(model, proj, current_task, args=model.args)
                    pred_task = torch.argmax(task_logits, dim=1)
                    #true_task_label= labels//2
                    #correct_tasks+=true_task_label==pred_task
                    logits_coomil = logits.clone()
                    mask_classes(logits_coomil, dataset, int(pred_task.item()))
                    Y_prob = F.softmax(logits_coomil, dim=1)
                    pred = torch.argmax(Y_prob, dim=1)

                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                prob_list.append(Y_prob.cpu().numpy())
                all_prob_list.append(Y_prob.cpu().numpy()[0][:2*num_task])
                labels_list.append(labels.item())
                all_labels_list.append(eye_array[labels.item()][:2*num_task])

                if dataset.SETTING == 'class-il':
                    mask_classes(logits, dataset, k)
                    _, mask_pred = torch.max(logits, 1)
                    mask_prob = F.softmax(logits, dim = 1)
                    mask_prob_list.append(mask_prob.cpu().numpy())
                    correct_mask_classes += torch.sum(mask_pred == labels).item()
        try:
            aucs.append(roc_auc_score(np.array(labels_list), np.concatenate(prob_list)[:, 2*k + 1]))
            aucs_mask_classes.append(roc_auc_score(np.array(labels_list) - (2*k), np.concatenate(mask_prob_list)[:, 2*k + 1]))
        except:
            print('Error in AUC calculation')
        
        accs.append(correct / total if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total)

        corrects += correct
        corrects_mask_classes += correct_mask_classes
        totals += total
    micro_acc = corrects / totals
    micro_acc_mask_classes = corrects_mask_classes / totals

    model.net.train(status)
    f1_score_val = f1_score(np.array(labels_list), [round(x) for x in np.concatenate(prob_list)[:, 2*k + 1]], average='weighted')
    if not last:
        try:
            all_aucs = roc_auc_score(np.array(all_labels_list), np.array(all_prob_list), multi_class='ovr')
        except:
            print('Error in AUC calculation')
            all_aucs = 0
        return [accs, micro_acc, accs_mask_classes, micro_acc_mask_classes, aucs, aucs_mask_classes, all_aucs]
    else:
        return [accs, micro_acc, accs_mask_classes, micro_acc_mask_classes, aucs, aucs_mask_classes]
        
def evaluate_val(model: ContinualModel, dataset: ContinualDataset, k, epoch, results_dir, early_stopping = None):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    # accs, accs_mask_classes = [], []
    prob_list, labels_list = [], []
    # for k, val_loader in enumerate(dataset.val_loader):
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    val_loss = 0
    val_loader= dataset.val_loaders[k]
    for data in val_loader:
        with torch.no_grad():
            # inputs, labels = data
            # inputs, labels = inputs.to(model.device), labels.to(model.device)
            # import ipdb;ipdb.set_trace()
            inputs0, inputs1, labels = data
            inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs0, k)
            else:
                outputs = model([inputs0, inputs1])

            logits, Y_prob, pred = outputs[:3]

            if model.NAME == 'cocoopmil_continual':
                    proj= outputs[-1]
                    current_task = model.current_task
                    _, task_logits = generate_task_logits(model, proj, current_task, args=model.args)
                    pred_task = torch.argmax(task_logits, dim=1)
                    #true_task_label= labels//2
                    #correct_tasks+=true_task_label==pred_task
                    logits_coomil = logits.clone()
                    mask_classes(logits_coomil, dataset, int(pred_task.item()))
                    Y_prob = F.softmax(logits_coomil, dim=1)
                    pred = torch.argmax(Y_prob, dim=1)

            val_loss += F.cross_entropy(logits, labels).item()
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            prob_list.append(Y_prob.cpu().numpy())
            labels_list.append(labels.item())

            if dataset.SETTING == 'class-il':
                # import ipdb;ipdb.set_trace()
                # mask_classes(outputs, dataset, k)
                mask_classes(logits, dataset, k)
                # _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
    # import ipdb;ipdb.set_trace()
    val_loss /= len(val_loader)
    auc = roc_auc_score(np.array(labels_list), np.concatenate(prob_list)[:, 2*k + 1])
    acc = correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0
    acc_mask_classes = correct_mask_classes / total * 100
    f1_score_val = f1_score(np.array(labels_list), [round(x) for x in np.concatenate(prob_list)[:, 2*k + 1]], average='weighted')
    model.net.train(status)
    print(f'\t auc = {auc}')
    wandb.log({"val/auc": auc,
               "val/acc": acc,
               "val/acc_mask_classes": acc_mask_classes,
               "val/loss": val_loss,
               "val/epoch": epoch,
               "val/f1_score": f1_score_val,
               })
    if early_stopping:
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, f"task{k}_checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False
    # return [acc, acc_mask_classes, auc]


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    acc_results, micro_acc_results, acc_results_mask_classes, micro_acc_results_mask_classes = [], [], [], []
    auc_results = []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args.test_fold, args.exp_desc)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)
    dataset_copy.load()
    for t in range(dataset.N_TASKS):
        model.net.train()
        #_, _, _ = dataset_copy.get_data_loaders(args.test_fold)
    
    dataset_copy.i = dataset.N_TASKS

    acc_random_results_class, micro_acc_random_results_class, acc_random_results_task, micro_acc_random_results_task, auc_random_results_class, _, all_auc_random_results_class = evaluate(model, dataset_copy)
    print(f'Random AUC = {all_auc_random_results_class}')
    print(file=sys.stderr)

    if 'joint' not in model.NAME:
        for t in range(dataset.N_TASKS):
            early_stopping = EarlyStopping(patience=10, stop_epoch=10, verbose=True, start_epoch=10)

            results_dir = f'./checkpoints/{args.exp_desc}'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            model.net.train()
            train_loader, val_loader, test_loader = dataset.get_data_loaders(args.test_fold)
            if hasattr(model, 'begin_task'):
  
                model.begin_task(train_loader)
            #if t:
            #    accs = evaluate(model, dataset, last=True)
#
            #    acc_results[t-1] = acc_results[t-1] + accs[0]
            #    auc_results[t-1] = auc_results[t-1] + accs[2]
            #    if dataset.SETTING == 'class-il':
            #        acc_results_mask_classes[t-1] = acc_results_mask_classes[t-1] + accs[2]
            #        # auc_results_mask_classes[t-1] = auc_results_mask_classes[t-1] + accs[3]

            scheduler = dataset.get_scheduler(model, args)


            if not args.debug_mode:
                if args.model == 'conslide' or args.model == 'derpp':
                    if t == 0:
                        for epoch in range(10):
                            loss = 0
                            for i, data in enumerate(train_loader):
                                inputs0, inputs1, labels = data
                                inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                                loss += model.observe(inputs0, inputs1, labels, t, ssl=True)

                                progress_bar(i, len(train_loader), epoch, t, loss, args.test_fold)
                            print(f'SSL Loss: {loss}')

            for epoch in range(model.args.n_epochs):
                for i, data in enumerate(train_loader):
                    inputs0, inputs1, labels = data
                    inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                    if args.model == 'conslide' or args.model == 'derpp':
                        loss = model.observe(inputs0, inputs1, labels, t, ssl=False)
                    elif args.model == 'ewc_on' or args.model == 'lwf':
                        loss = model.observe(inputs0, inputs1, labels)
                    elif args.model == 'er_ace' or args.model == 'gdumb':
                        loss = model.observe(inputs0, inputs1, labels, inputs1)
                    elif 'cocoopmil' in args.model:
                        loss = model.observe(inputs0, inputs1, labels)
                        #loss = observe_task(args, dataset, model, t, train_loader)

                    progress_bar(i, len(train_loader), epoch, t, loss, args.test_fold)

                    if args.tensorboard:
                        tb_logger.log_loss(loss, args, epoch, t, i)
                stop = evaluate_val(model, dataset, t, epoch=epoch, results_dir=results_dir, early_stopping=early_stopping)
                if stop:
                    break
                if scheduler is not None:
                    scheduler.step()
            if os.path.exists(os.path.join(results_dir, f"task{t}_checkpoint.pt")):
                model.load_state_dict(torch.load(os.path.join(results_dir, f"task{t}_checkpoint.pt")))

            # Add buffer data
            if args.save_buffer:
                if hasattr(model, 'save_buffer'):
                    
                
                    for i, data in enumerate(train_loader):
                        inputs0, inputs1, labels = data
                        inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                        model.save_buffer(inputs0, inputs1, labels, t)
            else:
                print('Buffer not saved, using the model without buffer')
            if hasattr(model, 'end_task'):
                model.end_task(train_loader)

            accs = evaluate(model, dataset)

            acc_results.append(accs[0])
            micro_acc_results.append(accs[1])
            auc_results.append(accs[5])
            acc_results_mask_classes.append(accs[2])
            micro_acc_results_mask_classes.append(accs[3])
            # auc_results_mask_classes.append(accs[3])
            print('\n')
            print(f'acc:                {accs[0]}')
            print(f'macro acc:          {np.mean(accs[0])}')
            print(f'micro acc:          {accs[1]}')
            print(f'mask acc:           {accs[2]}')
            print(f'macro mask acc:     {np.mean(accs[2])}')
            print(f'micro mask acc:     {accs[3]}')
            print(f'auc:                {accs[5]}')
            print(f'multi-classes auc:  {accs[6]}')
#            print(f'f1_score:           {accs[7]}')
            print('\n')


            mean_acc = np.mean(accs[0])
            micro_acc = accs[1]
            mean_acc_mask = np.mean(accs[2])
            micro_acc_mask = accs[3]
            mean_auc = np.mean(accs[5])
            multi_class_auc = accs[6]
            wandb.log({"test/mean_auc": mean_auc,"test/mean_acc": mean_acc, "test/multi_class_auc": multi_class_auc, "test/micro_acc": micro_acc, "test/mean_acc_mask": mean_acc_mask, "test/micro_acc_mask_classes": micro_acc_mask})
        
            # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

            if args.csv_log:
                csv_logger.log(mean_acc, mean_acc_mask, micro_acc, micro_acc_mask, mean_auc, multi_class_auc)
            if args.tensorboard:
                tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
    else:
        #_, _, _ = dataset.get_joint_data_loaders(args.test_fold)

        early_stopping = EarlyStopping(patience=20, stop_epoch=20, verbose=True, start_epoch=20)
        #early_stopping = EarlyStopping(patience=5, stop_epoch=50, verbose=True,start_epoch=int((3/5)*self.args.n_epochs))

        results_dir = f'./checkpoints/{args.exp_desc}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        model.net.train()
        train_loader, val_loader, test_loader = dataset.get_joint_data_loaders(args.test_fold)
        
        #scheduler = dataset.get_scheduler(model, args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.opt, args.n_epochs, 0.000005)

        for epoch in range(model.args.n_epochs):
            for i, data in enumerate(train_loader):
                inputs0, inputs1, labels = data
                inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                loss = model.observe(inputs0, inputs1, labels)                    

                progress_bar(i, len(train_loader), epoch, t, loss, args.test_fold)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
            stop = evaluate_val(model, dataset, t, epoch=epoch, results_dir=results_dir, early_stopping=early_stopping)
            if stop:
                break
            if scheduler is not None:
                scheduler.step()
        if os.path.exists(os.path.join(results_dir, f"task{t}_checkpoint.pt")):
            model.load_state_dict(torch.load(os.path.join(results_dir, f"task{t}_checkpoint.pt")))

        #if hasattr(model, 'end_task'):
        #    model.end_task(dataset, args.test_fold)
        
        accs = evaluate(model, dataset)

        acc_results.append(accs[0])
        micro_acc_results.append(accs[1])
        auc_results.append(accs[5])
        acc_results_mask_classes.append(accs[2])
        micro_acc_results_mask_classes.append(accs[3])
        # auc_results_mask_classes.append(accs[3])
        print('\n')
        print(f'acc:                {accs[0]}')
        print(f'macro acc:          {np.mean(accs[0])}')
        print(f'micro acc:          {accs[1]}')
        print(f'mask acc:           {accs[2]}')
        print(f'macro mask acc:     {np.mean(accs[2])}')
        print(f'micro mask acc:     {accs[3]}')
        print(f'auc:                {accs[5]}')
        print(f'multi-classes auc:  {accs[6]}')
        print('\n')


        mean_acc = np.mean(accs[0])
        micro_acc = accs[1]
        mean_acc_mask = np.mean(accs[2])
        micro_acc_mask = accs[3]
        mean_auc = np.mean(accs[5])
        multi_class_auc = accs[6]
        wandb.log({"test/mean_auc": mean_auc,"test/mean_acc": mean_acc, "test/multi_class_auc": multi_class_auc, "test/micro_acc": micro_acc, "test/mean_acc_mask": mean_acc_mask, "test/micro_acc_mask_classes": micro_acc_mask})
        
            
        # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if args.csv_log:
            csv_logger.log(mean_acc, mean_acc_mask, micro_acc, micro_acc_mask, mean_auc, multi_class_auc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:

        csv_logger.add_bwt(acc_results, acc_results_mask_classes, auc_results)
        csv_logger.add_forgetting(acc_results, acc_results_mask_classes, auc_results)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
           csv_logger.add_fwt(acc_results, acc_random_results_class,
                              acc_results_mask_classes, acc_random_results_task,
                              auc_results, auc_random_results_class)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))


#def observe_task(args, dataset, model, t, train_loader):
#    if args.debug_mode:
#        args.n_steps = 1
#    scheduler = dataset.get_scheduler(model, args)
#    #early_stopping = EarlyStopping(patience=25, stop_epoch=50, verbose=True, start_epoch=25)
#    early_stopping = EarlyStopping(patience=10, stop_epoch=10, verbose=True, start_epoch=10)
#    for epoch in range(model.args.n_epochs):
#        steps = 0
#        while steps < args.n_steps:
#            print(steps)
#            for i, data in enumerate(train_loader):
#                steps += 1
#                if steps > args.n_steps:
#                    break
#                inputs0, inputs1, labels = data
#                inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(
#                    model.device)
#                loss = model.meta_observe(inputs0, inputs1, labels)
#
#        stop = evaluate_val(model, dataset, t, epoch=epoch, results_dir=wandb.run.dir, early_stopping=early_stopping)
#        if stop:
#            break
#        if scheduler is not None:
#            scheduler.step()