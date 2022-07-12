import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

class Visualization:

    @staticmethod
    def gradient_visualization(cost, title):
        solvers, l_rates = cost.index.levels
        fig, axs = plt.subplots(1, len(solvers), sharey=True, figsize=(12, 6))
        fig.suptitle(title, fontsize=20)

        for idx, solver in enumerate(solvers):
            axs[idx].set_title(solver, fontsize=20)
            for rate in l_rates:
                axs[idx].plot(cost.loc[solver, rate], label=rate, alpha=0.7)
            axs[idx].set_facecolor("#F4F7F8")
            axs[idx].legend()
            axs[idx].grid(axis="y")
        fig.supylabel("Cost")
        fig.supxlabel("Iter num x100")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def metrics_visualize(metrics, score, threshold):
        sb.heatmap(data=metrics[metrics[score] > threshold], cmap="YlGn", annot=True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def predict_random_sample(model, X, Y):
        random_sample = np.random.choice(X.shape[1])
        prediction = model.predict(X[:, random_sample:random_sample+1])
        prediction = "cat" if prediction == 1 else "not a cat"
        actual_value = "cat" if Y[:, random_sample] == 1 else "not a cat"
        plt.title(f"Predictions for solver {model.solver} and learning rate equal {model.learning_rate}"
                  f"\nAcctual: {actual_value}, predicted: {prediction}")
        plt.imshow(X[:, random_sample:random_sample+1].reshape(64, 64, 3))
        plt.show()



