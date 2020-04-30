from cpc import CPC_train
from config import Args
from pprint import pprint
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mc', '--model-config', dest='model_config',
                        type=str2bool, default='yes')
    parser.add_argument('-lr', '--learning-rate', dest='lr',
                        type=float, default=3e-4)
    args_parser = parser.parse_args()

    args = Args()

    print(args_parser)
    if not args_parser.model_config:
        args.lr = args_parser.lr
        args.configs = str(args_parser.lr)

    pprint(vars(Args))
    cpc = CPC_train(args)
    cpc.train()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    main()
