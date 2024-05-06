import argparse
import utils
import os, sys
import logging
import glob
import torch

def parser_loader():
    parser = argparse.ArgumentParser(description='AdaGLT')
    parser.add_argument('--total_epoch', type=int, default=500)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument("--retain_epoch", type=int, default=300)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--spar_wei", default=False, action='store_true')
    parser.add_argument("--spar_adj", default=False, action='store_true')
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',)
    parser.add_argument('--save', type=str, default='CKPTs',
                        help='experiment name')
    parser.add_argument("--target_adj_spar", type=int, default=22) # 21-22
    parser.add_argument("--target_wei_spar", type=int, default=90)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--e1", type=float, default=2e-6)
    parser.add_argument("--e2", type=float, default=2e-3)
    parser.add_argument("--coef", type=float, default=0.01)

    args = vars(parser.parse_args())
    seed_dict = {'cora': 3999, 'citeseer': 19889, 'pubmed': 3333}
    # seed_dict = {'cora': 23977/23388, 'citeseer': 27943/27883, 'pubmed': 3333}
    args['seed'] = seed_dict[args['dataset']]
    torch.cuda.device(int(args['device'][-1]))

    if args['dataset'] == "cora":
        args['embedding_dim'] = [1433, 512, 7]
    elif args['dataset'] == "citeseer":
        args['embedding_dim'] = [3703, 512, 6]
    elif args['dataset'] == "pubmed":
        args['embedding_dim'] = [500, 512, 3]
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
