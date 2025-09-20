import optuna
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

def get_objective(X_train, X_valid, y_train, y_valid):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # 树的数量
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),  # 学习率
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # 树的最大深度
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),  # 每棵树使用的样本比例
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),  # 每棵树使用的特征比例
        }

        model = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric='logloss',
            radom_state=42)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        preds = model.predict(X_valid)
        return accuracy_score(y_valid, preds)

    return objective

study = optuna.create_study(direction="maximize")
study.optimize(get_objective(X_train, X_valid, y_train, y_valid), n_trials=50)

print("✅ Best Accuracy:", study.best_trial.value)
print("✅ Best Parameters:", study.best_trial.params)

fig1 = plot_optimization_history(study)
fig1.show()

fig2 = plot_param_importances(study)
fig2.show()

fig3 = plot_parallel_coordinate(study)
fig3.show()

fig4 = plot_contour(study)
fig4.show()

fig5 = plot_slice(study)
fig5.show()

fig1.write_html("optimization_history.html")
fig2.write_html("param_importances.html")
fig3.write_html("parallel_coordinate.html")
fig4.write_html("contour.html")
fig5.write_html("slice_plot.html")