from cpc import CPC_train
from config import Args
from pprint import pprint


def main():
    args = Args()
    pprint(vars(Args))
    cpc = CPC_train(args)
    cpc.train()


if __name__ == "__main__":
    main()