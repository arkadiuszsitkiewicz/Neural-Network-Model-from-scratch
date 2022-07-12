from src.data.data_management import DataManagement
from src.NN_model.NN_Model import NN_Model
from src.NN_model.basic_metrics import basic_metrics
from src.vis.visualize import Visualization
import pandas as pd
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


train_x, train_y, test_x, test_y = DataManagement.load_dataset("data/raw")
layers_dims = [20, 7, 5, 1]
solvers = ["relu", "sigmoid", "tanh"]
rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

# todo train and save models - it takes about 3-4min to learn one model
#  learned models are already provided in data/final_models
# for solver in solvers:
#     for rate in rates:
#         model = NN_Model(layers_dims, num_iteration=3000,  print_cost=False, solver=solver, learning_rate=rate)
#         params, cost = model.fit(train_x, train_y)
#         DataManagement.save_model("data/final_models", params, cost, solver, rate)

exported_models = None
for solver in solvers:
    for rate in rates:
        if exported_models is None:
            exported_models = pd.DataFrame(columns=["params", "cost"], index=[[solver], [rate]])
        exported_models.loc[(solver, rate), :] = DataManagement.load_model("data/final_models", solver, rate)

Visualization.gradient_visualization(exported_models.cost,
                                     "Gradient visualization for different solvers and learning rates")

metrics = None
for solver in solvers:
    for rate in rates:
        temp_model = NN_Model(layers_dims, solver=solver)
        temp_model.upload_params(exported_models.params.loc[solver, rate])
        preds = temp_model.predict(test_x)
        probas = temp_model.predict_proba(test_x)
        acc_metrics = basic_metrics(test_y, preds, probas)
        if metrics is None:
            metrics = pd.DataFrame(columns=acc_metrics.keys(), index=[[solver], [rate]], dtype=float)
        metrics.loc[(solver, rate), :] = acc_metrics

Visualization.metrics_visualize(metrics, "f1", 0.5)


best_solver, best_learning_rate = metrics.f1.idxmax()
best_params, _ = DataManagement.load_model("data/final_models", best_solver, best_learning_rate)
best_model = NN_Model(layers_dims, solver=best_solver, learning_rate=best_learning_rate)
best_model.upload_params(best_params)

Visualization.predict_random_sample(best_model, test_x, test_y)
