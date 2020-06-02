from mask_utils import MaskUtils
from os_utils import OSUtils

from torch import optim, nn, load


def train(model, train_dl, valid_dl, test_dl, args):
    """
    Trains a network iteratively according to the lottery ticket hypothesis algorithm. Depends on the function
    train_once

    Arguments
    --------
    model: (nn.Module) A feed forward model instance to train.
    train_dl: (DataLoader) The data loader of the training set
    valid_dl: (DataLoader) The data loader of the validation set
    test_dl: (DataLoader) The data loader of the test set
    args: (Namespace) The hyper-parameters of the training schedule as well as other relevant arguments.
    """
    print(f"training on {len(train_dl.dataset)}, validation {len(valid_dl.dataset)}, test {len(test_dl.dataset)}")
    model.to(args.device)
    mask = MaskUtils()
    mask.init_mask(model)

    # save the initial state_dict
    args.global_rate = f"rate_{int(args.rate * 100)}"
    initial_weights_path = f"{args.base_path}/{args.dataset}/{args.model_type}/{args.global_rate}/{args.seed}"

    if args.use_existing_state:
        model.load_state_dict(load(f"{args.base_path}/{initial_weights_path}/initial_weights.pt"))
    else:
        OSUtils.save_torch_object(model.state_dict(), initial_weights_path, "initial_weights.pt")

    num_weights = MaskUtils.num_weights(model)
    rate = 0
    initial_weights_path = f"{initial_weights_path}/initial_weights.pt"
    for iteration in range(args.start_iter, args.num_iterations):

        # first iteration is the complete dense network
        # all other iterations are pruned per the given rate
        if iteration != 0:
            rate = args.rate * (1 - rate) + rate
            mask.prune_network(model, rate)
            mask.reset_weights(model, initial_weights_path)

        num_active = MaskUtils.num_active(model)
        percentage_active = num_active / num_weights
        args.filename = f"{percentage_active:.5}.pt"
        print(f"\nIteration {iteration + 1}/{args.num_iterations}: % of weights active: {percentage_active:.2%}\n")

        train_once(model, train_dl, valid_dl, test_dl, mask, args)


def train_once(model, train_dl, valid_dl, test_dl, mask_object, args):
    """
    Trains one iteration of the lottery ticket hypothesis algorithm. It serves as a subroutine for the the
    function train above.

    Arguments
    ----------
    model:
    train_dl: The data loader for the training set
    valid_dl: The data loader for the validation set
    test_dl: The data loader for the test set
    mask_object: The mask utilities object. See mask_utils.py for details
    args: (Namespace object)

    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # metrics for the round
    loss_test = []
    acc_test = []
    loss_val = []
    acc_val = []

    i = 0
    while i < args.iters:
        # training phase
        for inputs, targets in train_dl:
            model.train()
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            predictions = model(inputs.float())
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()

            # zero gradients of the masked (pruned) weights before sgd update
            mask_object.freeze(model, weight=False)
            optimizer.step()

            # for completeness (and a waste of compute resources) double down and zero pruned weights in case
            # zeroing the gradients weren't effective.
            mask_object.freeze(model, weight=True)

            # evaluate on test and validation sets at the checkpoints
            if i % args.checkpoint == 0:
                val_loss, val_acc = evaluate(model, valid_dl, criterion, args.device)
                test_loss, test_acc = evaluate(model, test_dl, criterion, args.device)
                print(f"[{i}/{args.iters} val_loss {val_loss:.4} val_acc {val_acc:.2%}]")

                # update metrics for the epoch
                loss_test.append(test_loss)
                acc_test.append(test_acc)
                loss_val.append(val_loss)
                acc_val.append(val_acc)

            # stick to the number of iterations (50K for Lenet300100 on MNIST, 20k for CONV2 on CIFAR10, etc.,)
            if i == args.iters:
                break

            i += 1

    # save metrics, masks and final weights
    metrics = {"test_loss": loss_test, "test_acc": acc_test, "val_loss": loss_val, "val_acc": acc_val}

    dirs = f"{args.base_path}/{args.dataset}/{args.model_type}/{args.global_rate}/{args.seed}/metrics/"
    OSUtils.save_torch_object(metrics, dirs, args.filename)

    dirs = f"{args.base_path}/{args.dataset}/{args.model_type}/{args.global_rate}/{args.seed}/masks/"
    OSUtils.save_torch_object(mask_object.weight_mask, dirs, args.filename)

    dirs = f"{args.base_path}/{args.dataset}/{args.model_type}/{args.global_rate}/{args.seed}/final_weights/"
    OSUtils.save_torch_object(model.state_dict(), dirs, args.filename)


def evaluate(model, test_dl, criterion, device):
    """
    Arguments
    --------
    model: (nn.Module) The network to evaluate
    test_dl: (DataLoader) The evaluation dataset
    criterion: (nn.CrossEntropyLoss) The relevant loss function. Typically categorical cross entropy for all experiments
    device: (string) The device to execute the computation.

    Return
    -----
    loss_val: (float) The average lost over all examples in test_dl
    acc_val: (float) The fraction of total predictions which are correct
    """
    model.to(device)
    model.eval()
    loss_val = 0.0
    acc_val = 0.0
    for inputs, targets in test_dl:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs.float())
        loss = criterion(predictions, targets)
        loss_val += loss.item()
        acc_val += targets.eq(predictions.argmax(dim=1)).sum().item()

    loss_val /= len(test_dl)
    acc_val /= len(test_dl.dataset)

    return loss_val, acc_val
