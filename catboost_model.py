import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from config import Config


class CatBoost(object):

    def __init__(self, target: str, loss_function='RMSE', cv_metric='test-RMSE-mean',
                 iterations=5000, learning_rate=0.002,
                 depth=8, l2_leaf_reg=3, colsample_bylevel=0.8, bagging_temperature=1, subsample=1,
                 random_strength=1, min_data_in_leaf=20,
                 boosting_type='Ordered', bootstrap_type='Bayesian', od_type='Iter', od_wait=20,
                 fold_count=5, seed=7):

        self.train_path = Config().project_path + 'data/train.pkl'
        self.test_path = Config().project_path + 'data/test.pkl'
        self.model_path = Config().project_path + 'saved_models/catboost_' + target + '.pkl'

        self.target = target
        self.loss_function = loss_function
        self.cv_metric = cv_metric

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.colsample_bylevel = colsample_bylevel
        self.bagging_temperature = bagging_temperature
        self.subsample = subsample
        self.random_strength = random_strength
        self.min_data_in_leaf = min_data_in_leaf

        self.boosting_type = boosting_type
        self.bootstrap_type = bootstrap_type
        self.od_type = od_type
        self.od_wait = od_wait

        self.fold_count = fold_count
        self.seed = seed

        self.features = []
        self.cat_features = []

    def params(self) -> dict:
        """"""

        params = {"loss_function": self.loss_function,
                  "iterations": self.iterations,
                  "learning_rate": self.learning_rate,
                  "depth": self.depth,
                  "l2_leaf_reg": self.l2_leaf_reg,
                  "colsample_bylevel": self.colsample_bylevel,
                  "random_strength": self.random_strength,
                  "min_data_in_leaf": self.min_data_in_leaf,
                  "boosting_type": self.boosting_type,
                  "bootstrap_type": self.bootstrap_type,
                  "od_type": self.od_type,
                  "od_wait": self.od_wait,
                  "verbose": False,
                  "allow_writing_files": False
                  }

        if self.bootstrap_type == "Bayesian":
            params["bagging_temperature"] = self.bagging_temperature

        elif self.bootstrap_type == "Bernoulli":
            params["subsample"] = self.subsample

        return params

    def regressor(self):
        """"""
        # bagging_temperature
        if self.bootstrap_type == "Bayesian":
            return CatBoostRegressor(loss_function=self.loss_function,
                                     iterations=self.iterations,
                                     learning_rate=self.learning_rate,
                                     depth=self.depth,
                                     l2_leaf_reg=self.l2_leaf_reg,
                                     colsample_bylevel=self.colsample_bylevel,
                                     bagging_temperature=self.bagging_temperature,
                                     random_strength=self.random_strength,
                                     min_data_in_leaf=self.min_data_in_leaf,
                                     boosting_type=self.boosting_type,
                                     bootstrap_type=self.bootstrap_type,
                                     od_type=self.od_type,
                                     od_wait=self.od_wait,
                                     verbose=-1,
                                     allow_writing_files=False)

        # subsample
        elif self.bootstrap_type == "Bernoulli":
            return CatBoostRegressor(loss_function=self.loss_function,
                                     iterations=self.iterations,
                                     learning_rate=self.learning_rate,
                                     depth=self.depth,
                                     l2_leaf_reg=self.l2_leaf_reg,
                                     colsample_bylevel=self.colsample_bylevel,
                                     subsample=self.subsample,
                                     random_strength=self.random_strength,
                                     min_data_in_leaf=self.min_data_in_leaf,
                                     boosting_type=self.boosting_type,
                                     bootstrap_type=self.bootstrap_type,
                                     od_type=self.od_type,
                                     od_wait=self.od_wait,
                                     verbose=-1,
                                     allow_writing_files=False)

        else:
            return CatBoostRegressor(loss_function=self.loss_function,
                                     iterations=self.iterations,
                                     learning_rate=self.learning_rate,
                                     depth=self.depth,
                                     l2_leaf_reg=self.l2_leaf_reg,
                                     colsample_bylevel=self.colsample_bylevel,
                                     random_strength=self.random_strength,
                                     min_data_in_leaf=self.min_data_in_leaf,
                                     boosting_type=self.boosting_type,
                                     bootstrap_type=self.bootstrap_type,
                                     od_type=self.od_type,
                                     od_wait=self.od_wait,
                                     verbose=-1,
                                     allow_writing_files=False)

    def regressor_cv_score(self, params: dict, x, y, cat_features: list):
        """"""
        cv_dataset = Pool(data=x,
                          label=y,
                          cat_features=cat_features)

        return cv(cv_dataset,
                  params,
                  fold_count=self.fold_count,
                  shuffle=True,
                  seed=self.seed)[self.cv_metric].tolist()[-1]

    def cross_val_score(self):
        """"""
        train = joblib.load(self.train_path)
        x, y = train.loc[:, self.features], train[self.target]

        return self.regressor_cv_score(params=self.params(),
                                       x=x,
                                       y=y,
                                       cat_features=self.cat_features)

    def validation(self):
        """
            Check quality on validation set
        """

        validation = joblib.load(self.test_path)

        x = validation.loc[:, self.features]
        y = validation[self.target]

        models_list = joblib.load(self.model_path)

        y_hat = np.mean([model.predict(x) for model in models_list], axis=0, dtype=np.float64)

        trees = [model.tree_count_ for model in models_list]

        print("Trees", self.target, trees)
        print('MSE: ', mean_squared_error(y, y_hat))

        return validation

    def save_model(self, is_with_validation=True):
        """"""
        train_set = joblib.load(self.train_path)

        if is_with_validation:
            validation = joblib.load(self.test_path)
            train_set = pd.concat([train_set, validation])

        kf = KFold(n_splits=self.fold_count, shuffle=True, random_state=self.seed)

        models = []
        for train_index, test_index in kf.split(train_set):
            fold_train, fold_test = train_set.iloc[train_index], train_set.iloc[test_index]

            x_train, y_train = fold_train.loc[:, self.features], fold_train[self.target]
            x_test, y_test = fold_test.loc[:, self.features], fold_test[self.target]

            # unfitted model
            model = self.regressor()

            model.fit(x_train, y_train,
                      cat_features=self.cat_features,
                      early_stopping_rounds=self.od_wait,
                      eval_set=(x_test, y_test),
                      verbose=False)

            # fitted model
            models.append(model)

        joblib.dump(models, self.model_path)

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            Load train model and make predictions on new data (or test).
        """

        models_list = joblib.load(self.model_path)

        data['value'] = np.mean([model.predict(data.loc[:, self.features]) for model in models_list], axis=0, dtype=np.float64)

        return data

    def make_predictions_test(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            Load train model and make predictions on new data.
            Leave features for eye test
        """

        models_list = joblib.load(self.model_path)

        data[self.target] = np.mean([model.predict(data.loc[:, self.features]) for model in models_list], axis=0, dtype=np.float64)

        data = data.sort_values(['id_match', 'ishome'])

        return data