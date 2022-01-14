import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from powerprediction.utils.data_reader import basic_argparser, read_matlab
from powerprediction.utils.data_utils import train_val_test_split
import optuna


def objective(trial):
    num_leaves = trial.suggest_int("num_leaves", 10, 80, log=True)
    # n_estimators = trial.suggest_int("n_estimators", 10, 700, log=True)
    n_estimators = 150
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
    
    gbm = lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,
                            silent=True,  n_jobs=-1, random_state=17)
    #num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1, silent=True, importance_type='split', **kwargs
    gbm.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="l2", early_stopping_rounds=7, verbose = False)
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_, silent=True, verbose = 0);
    #print("The mae of prediction is:", type(mean_absolute_error(y_test, y_pred)))
    mae=mean_absolute_error(y_test,y_pred);

    mse = mean_squared_error(y_test, y_pred)
    return mse;


if __name__ == "__main__":
    args = basic_argparser.parse_args();
    dataset = read_matlab(args.filename, args.dataset);
    x, y = dataset.load_data(window_size=0);

    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x, y);

    x_train = x_train.reshape(x_train.shape[0], -1);
    x_val = x_val.reshape(x_val.shape[0], -1);
    x_test = x_test.reshape(x_test.shape[0], -1);

    # nsga_sampler = optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler(population_size=300, seed=17)
    nsga_sampler = optuna.samplers.NSGAIISampler(population_size=300, seed=17)
    study = optuna.create_study(direction="minimize", sampler=nsga_sampler)
    study.optimize(objective, n_trials=20)
    #print(study.best_trial)
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("##############TRIALS################")
    df = study.trials_dataframe()
    print(df)
    print("####################################")

    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
