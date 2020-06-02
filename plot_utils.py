import matplotlib.pyplot as plt
from torch import load, tensor
from matplotlib.ticker import FuncFormatter, ScalarFormatter


def get_test_acc(model_seeds, base_path, file_name):
    """
    Returns the test accuracy of a round in the lt hypothesis experiment

    Arguments
    ---------
    model_seeds:  (List) Indices of the independently randomly initialized models. Usually 5 models
    base_path: (String) The path to the base folder containing the metrics
    file_name: (String) The filename of the round.

    Returns
    ------
    test_acc: (Dict) The average test accuracy and the deviations from (min, max)

    """
    test_acc = []
    for i in model_seeds:
        metrics_i = load(f"{base_path}/{i}/metrics/{file_name}")
        test_acc.append(metrics_i["test_acc"])

    # get the min, max and averages of the five
    test_acc_ = tensor(test_acc)
    avg = test_acc_.mean(dim=0)
    test_acc = {
        "avg": avg,
        "min_max": [(avg - test_acc_.min(dim=0)[0]).tolist(), (test_acc_.max(dim=0)[0] - avg).tolist()]
    }

    return test_acc


def early_stop_acc(model_seeds, base_path, file_name):
    """
    get the early stopping iteration and test accuracy at that accuracy

    Arguments
    --------
    model_seeds: (List) Indices of the independently randomly initialized models. Usually 5 models
    base_path: (String) The path to the base folder containing the metrics
    file_name: (String) The filename of the round.

    Returns
    ------
    early_stop_metrics: (Dict) The average early stop test accuracy and the deviations from (min, max).
                    Also includes the average early stop iteration and the deviations from (min, max)
    """
    test_acc = []
    early_stop_iter = []
    for i in model_seeds:
        metrics_i = load(f"{base_path}/{i}/metrics/{file_name}")
        loss_i = metrics_i["val_loss"]
        early_stop_i = loss_i.index(min(loss_i))
        test_acc.append(metrics_i["test_acc"][early_stop_i])
        early_stop_iter.append(early_stop_i)

    # get the min, max and averages of the five
    # test_acc_ = tensor(test_acc)
    avg_acc = tensor(test_acc).mean().item()
    avg_iter = tensor(early_stop_iter).float().mean().long().item() * 100

    early_stop_metrics = {
        "avg_acc": avg_acc,
        "avg_iter": avg_iter,
        "min_max_acc": [avg_acc - min(test_acc), max(test_acc) - avg_acc],
        "min_max_iter": [avg_iter - (100 * min(early_stop_iter)), (max(early_stop_iter) * 100) - avg_iter]
    }

    return early_stop_metrics


def thousands(x, pos):
    """A formatter for one of the plots. Use K for thousands """
    return '%dK' % (x * 1e-3) if x > 0 else 0


def plot_lt_ticket():
    """Plots the Fig 2 in the lottery ticket hypothesis."""
    model_seeds = [0, 1, 2, 3, 4]
    base_path = "./mnist/lenet300100/rate_20" # change to match your path

    test_acc_baseline = get_test_acc(model_seeds, base_path, file_name="1.0.pt")

    plt.rcParams.update({"font.size": 20})

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(42, 12))

    file_names = [["0.51292.pt", "0.2112.pt"], ["0.07047.pt", "0.036998.pt"], ["0.016262.pt"]]
    fmts = [["orange", "green"], ["red", "silver"], ["brown"]]
    iters = [i * 100 for i in range(len(test_acc_baseline["avg"]))]
    handles = []
    for i in range(len(file_names)):
        ax_i = axes[i]
        ax_i.set_xlim(left=0, right=20000)
        ax_i.set_ylim(bottom=0.94, top=0.99)
        ax_i.set_xticks([0, 5000, 10000, 15000])
        ax_i.set_yticks([0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
        ax_i.set_xlabel("Training Iterations")
        ax_i.set_ylabel("Test Accuracy")
        ax_i.grid(b=True)
        h_0 = ax_i.errorbar(
            iters, test_acc_baseline["avg"], yerr=test_acc_baseline['min_max'], fmt="b-", errorevery=12,
            capsize=2, capthick=2
        )
        if i == 0:
            handles.append(h_0)

        for j in range(len(file_names[i])):
            test_acc_i = get_test_acc(model_seeds, base_path, file_name=file_names[i][j])
            h_i = ax_i.errorbar(
                iters, test_acc_i["avg"], yerr=test_acc_i['min_max'], fmt="-",
                color=f"{fmts[i][j]}", errorevery=12)

            handles.append(h_i)

    figure.legend(handles=handles, labels=["100.0", "51.3", "21.1", "7.0", "3.6", "1.9"],
                  loc="upper center", ncol=6, fancybox=True)
    plt.savefig("lt_plot.png")


def plot_early_ticket():
    """Plots parts of Figure 1 of the lottery ticket hypothesis paper"""

    model_seeds = [0, 1, 2, 3, 4]
    base_path = "./mnist/lenet300100/rate_20" # change to match your path
    rates = [
        1.0, 0.80038, 0.64068, 0.51292, 0.41071, 0.32894, 0.26353, 0.2112, 0.16934, 0.13585,
        0.10905, 0.087618, 0.07047, 0.056747, 0.045774, 0.036998, 0.02997, 0.024354, 0.019857, 0.016262,
        0.013388, 0.011086, 0.0092412, 0.0077686, 0.0065928, 0.0056499, 0.0048986, 0.0042938, 0.0038092, 0.0034222
    ]

    early_stopping_metrics = [early_stop_acc(model_seeds, base_path, f"{r}.pt") for r in rates]

    acc_avg = [m["avg_acc"] for m in early_stopping_metrics]
    acc_error_bars = tensor([m["min_max_acc"] for m in early_stopping_metrics]).T
    iters_avgs = [m["avg_iter"] for m in early_stopping_metrics]
    iter_error_bars = tensor([m["min_max_iter"] for m in early_stopping_metrics]).T

    plt.rcParams.update({"font.size": 20})
    figure, (axe1, axe2) = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))
    formatter = FuncFormatter(thousands)
    axe1.set_xlim(105, -5)
    axe2.set_xlim(105, -5)
    x_ticks = [100, 41.1, 16.9, 7.0, 2.9, 1.2, 0.5, 0.3]
    iters = [round(i * 100, 1) for i in rates]
    axe1.set_xscale("log")
    axe2.set_xscale("log")
    axe1.set_xticks(x_ticks)
    axe2.set_xticks(x_ticks)

    axe1.set_xlabel("Percentage of Weights Remaining")
    axe2.set_xlabel("Percentage of Weights Remaining")
    axe1.set_ylabel("Early Stop Iteration (Val)")
    axe2.set_ylabel("Accuracy at Early Stop (Test)")

    axe1.set_ylim(-10, 40000)
    axe1.set_yticks([0, 20000, 40000])
    axe1.yaxis.set_major_formatter(formatter)
    axe1.xaxis.set_major_formatter(ScalarFormatter())
    axe2.xaxis.set_major_formatter(ScalarFormatter())
    axe2.set_yticks([0.95, 0.96, 0.97, 0.98, 0.99])
    axe2.set_ylim(0.95, 0.99)

    axe1.errorbar(iters, iters_avgs, yerr=iter_error_bars, fmt="r-")
    axe2.errorbar(iters, acc_avg, yerr=acc_error_bars, fmt="r-")

    axe1.grid(b=True)
    axe2.grid(b=True)

    plt.suptitle("(MNIST, Lenet300100, LT, r = 20% )")
    plt.savefig("early_stop.png")


# plots
plot_early_ticket()
plt.clf()
plot_lt_ticket()
