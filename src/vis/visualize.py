import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


class Visualization:

    @staticmethod
    def gradient_visualization(cost, cost_step, title):
        solvers, l_rates = cost.index.levels
        fig, axs = plt.subplots(1, len(solvers), sharey=True, figsize=(12, 6))
        fig.suptitle(title, fontsize=20)

        if len(solvers) != 1:
            for idx, solver in enumerate(solvers):
                axs[idx].set_title(solver, fontsize=20)
                for rate in l_rates:
                    axs[idx].plot(cost.loc[solver, rate], label=rate, alpha=0.7)
                axs[idx].set_facecolor("#F4F7F8")
                axs[idx].legend()
                axs[idx].grid(axis="y")
                axs[idx].set_ylim([0, 1])

        else:
            axs.set_title(solvers[0], fontsize=20)
            for rate in l_rates:
                axs.plot(cost.loc[solvers[0], rate], label=rate, alpha=0.7)
            axs.set_facecolor("#F4F7F8")
            axs.legend()
            axs.grid(axis="y")
        fig.supylabel("Cost")
        fig.supxlabel(f"Iter num x{cost_step}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def metrics_visualize(metrics, score="f1", threshold=0):
        sb.heatmap(data=metrics[metrics[score] >= threshold], cmap="YlGn", annot=True)
        plt.yticks(rotation=0)
        plt.show()

    @staticmethod
    def predict_random_sample(model, X, Y):
        random_sample = np.random.choice(X.shape[1])
        prediction = model.predict(X[:, random_sample:random_sample+1])
        prediction = "cat" if prediction == 1 else "not a cat"
        actual_value = "cat" if Y[:, random_sample] == 1 else "not a cat"
        plt.title(f"Predictions for activation function {model.activation} and learning rate {model.learning_rate}"
                  f"\nAcctual: {actual_value}, predicted: {prediction}")
        plt.imshow(X[:, random_sample:random_sample+1].reshape(64, 64, 3))
        plt.show()

    @staticmethod
    def plot_decision_boundary(predict_func, x, y):
        x_min, x_max = x[0, :].min() - 1, x[0, :].max() + 1
        y_min, y_max = x[1, :].min() - 1, x[1, :].max() + 1
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = predict_func(np.c_[xx.flatten(), yy.flatten()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm)
        plt.ylabel("x2")
        plt.xlabel("x1")
        plt.scatter(x[0, :], x[1, :], c=y, cmap=plt.cm.coolwarm)
        plt.show()
