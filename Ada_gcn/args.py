import argparse
import utils
import os, sys
import logging
import glob

def parser_loader():
    parser = argparse.ArgumentParser(description='AdaGLT')
    parser.add_argument('--total_epoch', type=int, default=400)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument("--retain_epoch", type=int, default=300)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:6")

    parser.add_argument("--spar_wei", default=False, action='store_true')
    parser.add_argument("--spar_adj", default=False, action='store_true')
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',)
    parser.add_argument('--save', type=str, default='CKPTs',
                        help='experiment name')
    parser.add_argument("--target_adj_spar", type=int, default=27) # 21-22
    parser.add_argument("--target_wei_spar", type=int, default=93)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--use_bn", action="store_true", default=False)
    parser.add_argument("--use_res", action="store_true", default=False)
    parser.add_argument("--e1", type=float, default=5e-5)
    parser.add_argument("--e2", type=float, default=1e-3)
    parser.add_argument("--coef", type=float, default=0.1)
    parser.add_argument("--task_type", type=str, default="semi")

    args = vars(parser.parse_args())
    seed_dict = {'cora': 1899, 'citeseer': 17889, 'pubmed': 3333}
    # seed_dict = {'cora': 23977/23388, 'citeseer': 27943/27883, 'pubmed': 3333}
    args['seed'] = seed_dict[args['dataset']]

    if args['dataset'] == "cora":
        args['embedding_dim'] = [1433,] +  [512,] * (args['num_layers'] - 1) + [7]
    elif args['dataset'] == "citeseer":
        args['embedding_dim'] = [3703,] +  [512,] * (args['num_layers'] - 1) + [6]
    elif args['dataset'] == "pubmed":
        args['embedding_dim'] = [500,] +  [512,] * (args['num_layers'] - 1) + [3]
    else:
        raise NotImplementedError("dataset not supported.")

    args["model_save_path"] = os.path.join(
        args["save"], args["model_save_path"])
    utils.create_exp_dir(args["save"], scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    return args
