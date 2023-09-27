import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score


class CatBoost(object):

    def __init__(self, target: str, is_regression=True, loss_function='RMSE', cv_metric='test-RMSE-mean',
                 iterations=1000, learning_rate=0.03,
                 depth=8, l2_leaf_reg=3, colsample_bylevel=0.8, bagging_temperature=1, subsample=0.66,
                 random_strength=1, min_data_in_leaf=20,
                 boosting_type='Ordered', bootstrap_type='Bayesian', od_type='Iter', od_wait=20,
                 fold_count=5, seed=7):

        self.is_regression = is_regression

        self.train_path = ''
        self.validation_path = ''
        self.test_path = ''

        self.model_path = ''

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
                                     verbose=False,
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
                                     verbose=False,
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
                                     verbose=False,
                                     allow_writing_files=False)

    def classifier(self):
        """"""
        # bagging_temperature
        if self.bootstrap_type == "Bayesian":
            return CatBoostClassifier(loss_function=self.loss_function,
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
                                      verbose=False,
                                      allow_writing_files=False)

        # subsample
        elif self.bootstrap_type == "Bernoulli":
            return CatBoostClassifier(loss_function=self.loss_function,
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
                                      verbose=False,
                                      allow_writing_files=False)

        else:
            return CatBoostClassifier(loss_function=self.loss_function,
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
                                      verbose=False,
                                      allow_writing_files=False)

    def optuna_cv_score(self, params: dict, cv_dataset: Pool) -> float:
        """"""
        return cv(cv_dataset,
                  params,
                  fold_count=self.fold_count,
                  shuffle=False,
                  seed=self.seed)[self.cv_metric].tolist()[-1]

    def cross_val_score(self):
        """"""
        train = pd.read_feather(self.train_path)
        x, y = train.loc[:, self.features], train[self.target]

        cv_dataset = Pool(data=x,
                          label=y,
                          cat_features=self.cat_features)

        return cv(cv_dataset,
                  self.params(),
                  fold_count=self.fold_count,
                  shuffle=True,
                  seed=self.seed)[self.cv_metric].tolist()[-1]

    def validation(self) -> pd.DataFrame:
        """
            Check the model quality on the validation set.
        """

        validation = pd.read_feather(self.test_path)

        x = validation.loc[:, self.features]
        y = validation[self.target]

        if self.is_regression:
            model = CatBoostRegressor()
            model.load_model(self.model_path)
            y_hat = model.predict(x)

            print('MSE: ', mean_squared_error(y, y_hat))
            print('MAE: ', mean_absolute_error(y, y_hat))
        else:
            model = CatBoostClassifier()
            model.load_model(self.model_path)
            y_hat = model.predict_proba(x)

            print('LogLoss: ', log_loss(y, y_hat))
            print('Accuracy: ', accuracy_score(y, y_hat.round()))

        return validation

    def save_model(self):
        """"""
        train_set = pd.read_feather(self.train_path)
        validation_set = pd.read_feather(self.test_path)

        x_train, y_train = train_set.loc[:, self.features], train_set[self.target]
        x_val, y_val = validation_set.loc[:, self.features], validation_set[self.target]

        # unfitted model
        if self.is_regression:
            model = self.regressor()
        else:
            model = self.classifier()

        model.fit(x_train, y_train,
                  cat_features=self.cat_features,
                  early_stopping_rounds=self.od_wait,
                  eval_set=(x_val, y_val),
                  verbose=False)

        model.save_model(self.model_path)

    def make_predictions(self, data: pd.DataFrame, prediction_name: str = None) -> pd.DataFrame:
        """
            Loads trained model and makes predictions on the new data (or test).
        """

        if prediction_name is None:
            prediction_name = 'prediction'

        model = CatBoostRegressor()

        model.load_model(self.model_path)

        data[prediction_name] = model.predict(data.loc[:, self.features])

        return data
