from cpc import CPC_train
from config import Args


def main():
    args = Args()
    cpc = CPC_train(args)
    cpc.train()


if __name__ == "__main__":
    main()