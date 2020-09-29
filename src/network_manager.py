import numpy as np
from networks.matrix_neural_net import NeuralNetwork as MatrixNeuralNetwork
from networks.oop_neural_net import NeuralNetwork as OOPNeuralNetwork
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


class NetworkManager:
    def __init__(self, network):
        self.network = network

    def train_test(
        self,
        X,
        y,
        mode="classify",
        test_split=0.2,
        learning_rate=None,
        train_indices=None,
        test_indices=None,
        progress=None,
    ):
        """
        progress is a callback function that will be invoked with:
            progress(b:int, tsize:int)
                b: current state of progress (ie: iteration)
                tsize: final state of progress (ie: total)
        """
        self.score = self.score_classification

        if mode == "regress":
            self.score = self.score_regression

        if not train_indices and not test_indices:

            idx = np.arange(len(X))
            np.random.shuffle(idx)
            test_indices = idx[: int(test_split * len(X))]
            train_indices = idx[int(test_split * len(X)) :]

        # training phase
        training = []
        for i, tr_index in enumerate(train_indices):
            network_input = X[tr_index]
            output = self.network.forward_pass(network_input)
            expected = y[tr_index]

            if mode == "classify":
                training.append((np.argmax(expected), np.argmax(output)))
            else:
                training.append((expected[0], output[0]))

            if learning_rate == "step":
                step_size = int(len(train_indices) / 20)
                if not i % step_size and i > 0:
                    self.network.learning_rate = self.network.learning_rate / 2

            self.network.backward_pass(
                network_input=network_input,
                network_output=output,
                expected_output=expected,
            )
            if progress is not None:
                progress(b=i, tsize=len(train_indices))

        # testing phase
        testing = []
        for i, te_index in enumerate(test_indices):

            output = self.network.forward_pass(X[te_index])
            expected = y[te_index]

            if mode == "classify":
                testing.append((np.argmax(expected), np.argmax(output)))
            else:
                testing.append((expected[0], output[0]))

            if progress:
                progress(b=i, tsize=len(test_indices))

        return self.score(training), self.score(testing)

    def calculate_fscore(self, precision: float, recall: float, beta: int = 1):
        return (
            (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        )

    def score_classification(self, results: List[Tuple]):
        max_val = max(list(sum(results, ())))

        confusion = np.zeros((max_val, max_val))
        for i, j in results:
            confusion[(i - 1, j - 1)] += 1

        precision = [confusion[(i, i)] / sum(confusion[i, :]) for i in range(max_val)]
        recall = [confusion[(i, i)] / sum(confusion[:, i]) for i in range(max_val)]

        accuracy = sum(int(actual == expected) for expected, actual in results) / len(
            results
        )

        fscore = self.calculate_fscore(np.mean(precision), np.mean(recall), beta=1)

        return {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "fscore": fscore,
        }

    def mean_squared_error(self, y, yhat):
        return (1 / len(y)) * sum((y - yhat) ** 2)

    def r_squared(self, y, yhat):
        ybar = sum(y) / len(y)
        return 1 - ((sum((y - yhat) ** 2) / sum((y - ybar) ** 2)))

    def score_regression(self, results: List[Tuple]):
        y = np.array([r[0] for r in results])
        yhat = np.array([r[1] for r in results])
        return {
            "mean_squared_error": self.mean_squared_error(y, yhat),
            "r2": self.r_squared(y, yhat),
        }

    # KFold cross validation for hyper paramter tuning
    def kfold(self, X, y, k, split=0.2, mode="classify", progress=None):
        testing = []
        training = []

        fold_size = int(len(X) * split)
        if int(len(X) / k) < fold_size:
            fold_size = int(len(X) / k)

        idx = np.arange(len(X))
        np.random.shuffle(idx)
        idx = list(idx)

        for i in range(k):

            test_idx = idx[i * fold_size : (i + 1) * fold_size]
            train_idx = idx[: i * fold_size]
            train_idx.extend(idx[(i + 1) * fold_size :])

            train_score, test_score = self.train_test(
                X,
                y,
                test_indices=test_idx,
                train_indices=train_idx,
                mode=mode,
            )
            testing.append(test_score)
            training.append(train_score)

            if progress is not None:
                progress(b=i, tsize=k)

        return training, testing

    # def train_test_minibatch(
    #     self,
    #     X,
    #     y,
    #     mode="classify",
    #     learning_rate=None,
    #     test_split=0.2,
    #     batch_size=32,
    #     epochs=100,
    #     progress=None,
    # ):
    #     """
    #     progress is a callback function that will be invoked with:
    #         progress(current:int, total:int)
    #     """

    #     # training phase

    #     train_accuracies = []
    #     test_accuracies = []

    #     for epoch in range(epochs):

    #         idx = np.arange(len(X))
    #         np.random.shuffle(idx)
    #         test_indices = idx[: int(test_split * len(X))]
    #         train_indices = idx[int(test_split * len(X)) :]

    #         back_prop_inputs = []
    #         correct = 0
    #         for i, tr_index in enumerate(train_indices):

    #             network_input = X[tr_index]
    #             output = self.network.forward_pass(network_input)
    #             expected = y[tr_index]

    #             if mode == "classify":
    #                 correct += int(np.argmax(output) == np.argmax(expected))

    #             back_prop_inputs.append(
    #                 {
    #                     "network_input": network_input,
    #                     "network_output": output,
    #                     "expected_output": expected,
    #                 }
    #             )

    #             if not i % batch_size:
    #                 for bp_input in back_prop_inputs:
    #                     self.network.backward_pass(**bp_input)
    #                 back_prop_inputs = []

    #         train_accuracies.append(correct / len(train_indices))

    #         if learning_rate == "step":
    #             step_size = int(epochs / 20)
    #             if not i % step_size and i > 0:
    #                 self.network.learning_rate = self.network.learning_rate / 2

    #         # testing phase
    #         correct = 0
    #         for _, te_index in enumerate(test_indices):

    #             output = self.network.forward_pass(X[te_index])
    #             expected = y[te_index]

    #             if mode == "classify":
    #                 correct += int(np.argmax(output) == np.argmax(expected))

    #         test_accuracies.append(correct / len(test_indices))

    #         # update progress after each epoch
    #         if progress:
    #             progress(b=epoch, tsize=epochs)

    #     return train_accuracies, test_accuracies


if __name__ == "__main__":

    # train against sin

    model_params = {
        "shape": [1, 10, 25, 1],
        "activation": "tanh",
        "output_activation": "tanh",
        "learning_rate": 0.05,
        "loss": "mean_squared_error",
    }
    oop_nn = OOPNeuralNetwork(**model_params)
    manager = NetworkManager(oop_nn)

    X = np.random.uniform(0, 1.0, 5000)
    y = np.sin(2 * 2 * np.pi * X)

    # input layer expects a list (even if it's just one element)
    X = [[i] for i in X]
    y = [[i] for i in y]

    # with TqdmUpTo(desc="Training OOP Neural Network against SIN(x) function") as t:
    #     train_oop, test_oop = manager.kfold(
    #         X, y, 50, mode="regress", progress=t.update_to
    #     )

    matrix_nn = MatrixNeuralNetwork(**model_params)
    manager = NetworkManager(matrix_nn)

    with TqdmUpTo(desc="Training Matrix Neural Network against SIN(x) function") as t:
        train_matrix, test_matrix = manager.kfold(
            X, y, 20, mode="regress", progress=t.update_to
        )

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title(
        "MSE in OOP Neural Network",
        fontsize=6,
    )
    # axs[0, 0].plot(
    #     [r["mean_squared_error"] for r in train_oop], "tab:orange", label="Training"
    # )
    # axs[0, 0].plot(
    #     [r["mean_squared_error"] for r in test_oop], "tab:blue", label="Testing"
    # )
    # axs[0, 0].legend()
    # axs[0, 1].set_title(
    #     "R2 in OOP Neural Network",
    #     fontsize=6,
    # )
    # axs[0, 1].plot([r["r2"] for r in train_oop], "tab:orange", label="Training")
    # axs[0, 1].plot([r["r2"] for r in test_oop], "tab:blue", label="Testing")
    axs[0, 1].legend()
    axs[1, 0].set_title(
        "MSE in Matrix Neural Network",
        fontsize=6,
    )
    axs[1, 0].plot(
        [r["mean_squared_error"] for r in train_matrix], "tab:orange", label="Training"
    )
    axs[1, 0].plot(
        [r["mean_squared_error"] for r in test_matrix], "tab:blue", label="Testing"
    )
    axs[1, 0].legend()
    axs[1, 1].set_title(
        "R2 in Matrix Neural Network",
        fontsize=6,
    )
    axs[1, 1].plot([r["r2"] for r in train_matrix], "tab:orange", label="Training")
    axs[1, 1].plot([r["r2"] for r in test_matrix], "tab:blue", label="Testing")
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

    # visual = input("View sample predictions? (y/n)\n")
    # if visual == "y":
    X = np.arange(0, 1, 0.001)
    yhat = np.sin(2 * 2 * np.pi * X)
    y = [matrix_nn.forward_pass([x]) for x in X]

    plt.plot(X, yhat, "tab:gray")
    plt.scatter(X, [i[0] for i in y], c="r")
    plt.show()
