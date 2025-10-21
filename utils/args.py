# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--exp_desc', type=str, required=True,
                        help='Experiment description.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    

    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')
    

    #COOMIL#####################################################

    parser.add_argument('--exclude_from_optimizer', type=str, nargs='+', default=["text_encoder"])
    parser.add_argument('--exclude_from_optimizer2', type=str, nargs='+',
                        default=["text_encoder", "dsmil", "meta_net", "prompt_learner"])
    parser.add_argument('--coomil_lr', type=float, default=0.0003, help='Learning rate.')
    parser.add_argument('--n_tasks_per_model', type=int, default=None, help='Number of tasks.')
    parser.add_argument('--coomil_optimizer', type=str, default="adam", choices=["adam", "sgd"],
                        help='optimizer weight decay.')
    parser.add_argument('--coomil_optim_wd', type=float, default=0.000001, help='optimizer weight decay.')
    parser.add_argument('--observe_task', default=1, choices=[0, 1], type=int, help='observe task')
    
    parser.add_argument('--n_steps', type=int, default=300, help='Batch size.')
    #parser.add_argument("--eval_logits", default=2, choices=[0,1,2], type=int)



def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging', default=True)
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


    #COOMIL#####################################################
    #parser.add_argument('--notes', type=str, default=None, help='Notes for this run.')
    #parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')

    parser.add_argument('--dropout_patch', type=int, default=200, help='Crop size')
    parser.add_argument('--temperature', type=int, default=20, help='Wandb project name')
    parser.add_argument('--dsmil_freezed', default=0, choices=[0, 1], type=int, help='disable additional metrics')
    parser.add_argument('--test_on_val', default=0, choices=[0, 1], type=int, help='Test on val')

    parser.add_argument('--n_classes', default=8, type=int, help='Number of output classes [2]')
    parser.add_argument('--n_classes_per_task', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--c_hidden', default=256, type=int, help='intermediate size ')
    parser.add_argument('--input_size', default=512, type=int, help='input size ')
   
    parser.add_argument('--init_prompts', default=False)
    parser.add_argument('--softmaxnormal', default=True)
    parser.add_argument('--dsmil_multiscale', default=True)
    parser.add_argument('--dualcoop_multiscale', default=True)
    parser.add_argument('--context', default=True)
    parser.add_argument('--attenpool', default=False)
    parser.add_argument('--loss', default="ce")
    parser.add_argument('--bag_context', default=True)
    parser.add_argument('--add_context_only_learnable', default=False)
    parser.add_argument('--add_tumor_context', default=True)
    parser.add_argument('--add_normal_context', default=True)
    parser.add_argument('--use_prompt_templates', default=True)
    parser.add_argument('--loss_on_task_logits', default=False)
    parser.add_argument("--n_clusters", type=int, default=14)

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=False, default=100,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')