from models import construct_model
from train import train
from dataset_utils import get_dataset

from argparse import ArgumentParser
import torch


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, test_dl = get_dataset(args.batch_size, args.dataset)
    model = construct_model(args.model_type)
    train(model, train_dl, val_dl, test_dl, args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, help="The dataset being used for training")
    parser.add_argument("--model_type", default="lenet300100", type=str, help="The type (architecture) of the model")

    # training schedule
    parser.add_argument("--lr", default=1.2e-3, type=float, help="the learning rate")
    parser.add_argument("--batch_size", default=60, type=int, help="The batch size")
    parser.add_argument("--num_iterations", default=30, type=int, help="number of train-prune iterations")
    parser.add_argument("--iters", default=50000, type=int, help="Number of iterations for each pruning stage")
    parser.add_argument("--start_iter", default=0, type=int, help="The iteration number to resume")
    parser.add_argument("--seed", default=0, type=int, help="Index for a randomly seeded model. It is used mainly")
    parser.add_argument("--checkpoint", default=100, type=int, help="Validation and test checkpoint")
    parser.add_argument("--rate", default=0.2, type=float, help="The global prune rate")
    parser.add_argument("--use_existing_state", default=False, type=bool, help="Use a previously seeded model.")
    parser.add_argument("--base_path", default=".", type=str, help="The base path to save checkpoints and metrics")

    arguments = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    main(arguments)
