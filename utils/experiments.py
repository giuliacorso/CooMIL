import copy
import itertools

def get_experiments(args):
    args1 = copy.copy(args)
    args1.mem = 120
    args1.dataset = "seq-wsi"
    args1.exp_desc= args.model
    args1.alpha=0.2
    args1.seed = 12
    args1.beta=0.2
    hyperparameters = [[0,1,2,3,4,5,6,7,8,9],[5,6,7,8,9,0,1,2,3,4]]  
    args_list = []
    for element0, element1 in zip(*hyperparameters):
        args2 = copy.copy(args1)
        args2.test_fold = element0
        args2.val_fold = element1
        if args.debug_mode:
            args2.n_epochs=2
        else:
            args2.n_epochs=50
        args_list.append(args2)
    return args_list
