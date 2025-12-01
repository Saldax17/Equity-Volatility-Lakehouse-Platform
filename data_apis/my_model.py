import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib
import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from dateutil.relativedelta import relativedelta
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from pathlib import Path
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
import re
import matplotlib.cm as cm

class MyModel:
    def __init__(self, file_path: str, start_year = None):
        self.__df = pd.DataFrame()    
        self.__file_path = file_path
        self.__start_year = start_year

    def load_from_file(self, file_name: str = 'merged_data.csv', test_periods: int = 12):
        self.__df = pd.read_csv(f"{self.__file_path}/{file_name}")
        self.__organize_data()
        self.__test_periods = test_periods

    def load_from_df(self, df: pd.DataFrame, test_periods: int = 12):
        self.__df = df
        self.__organize_data()
        self.__test_periods = test_periods

    def load_from_organized_df(self, df: pd.DataFrame, test_periods: int = 12):
        self.__df = df
        self.__test_periods = test_periods

    def get_df(self) -> pd.DataFrame:
        return self.__df

    def __organize_data(self):
        self.__df['timestamp'] = pd.to_datetime(self.__df['timestamp'], utc=True)
        self.__df['timestamp'] = self.__df['timestamp'].dt.tz_convert('America/New_York') #type: ignore
        self.__df = self.__df.sort_values(['symbol','timestamp']).reset_index(drop=True)
        # Add a new column with False when market_absolute_path is less than 0.06 and True when greater than or equal to 0.06
        self.__df['market_path_flag'] = (self.__df['market_absolute_path'] >= 0.06)
        self.apply_one_hot_encoding(['weekday', 'size_index','sector_index'])

    def apply_one_hot_encoding(self, categorical_columns: list):
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(self.__df[categorical_columns])
        df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
        self.__df = pd.concat([self.__df, df_encoded], axis=1).drop(columns=categorical_columns)

    def trim_top_rows(self, n_rows: int):
        mask = self.__df.groupby('symbol').cumcount() >= n_rows
        self.__df = self.__df[mask].reset_index(drop=True)

    def fill_empty_dsh(self):
        # Fill up nans on days_since_holiday with the median of the column
        median_days_since_holiday = self.__df['days_since_holiday'].median()
        self.__df['days_since_holiday'] = self.__df['days_since_holiday'].fillna(median_days_since_holiday)

    def df_preparation(self,rows_to_trim: int = 20):
        self.trim_top_rows(n_rows=rows_to_trim)
        self.fill_empty_dsh()
        self.__df['date'] = self.__df['timestamp'].dt.date #type: ignore
        # Place 'date' column as the second column
        self.__df.insert(1, 'date', self.__df.pop('date'))
        # Set market_path_flag as int instead of bool
        self.__df['market_path_flag'] = self.__df['market_path_flag'].astype(int)
        # Delete unneeded columns for the model
        self.__df = self.__df.drop(columns=['market_absolute_path',
                                        'symbol',
                                        'timestamp',
                                        'open',
                                        'high',
                                        'low',
                                        'close',
                                        'close_adj',
                                        'volume',
                                        'trade_count',
                                        ])
        #Sort by date and reset index
        self.__df = self.__df.sort_values(['date']).reset_index(drop=True)
        
    def show_var_selection_results(self):
        pipeline_forward = joblib.load(f"{self.__file_path}\\pipeline_forward.joblib")
        pipeline_backward = joblib.load(f"{self.__file_path}\\pipeline_back.joblib")
        #Loop twice, once for forward and once for backward
        for pipeline, method in zip([pipeline_forward, pipeline_backward], ['Forward Selection', 'Backward Elimination']):
            model = pipeline.named_steps['classifier']
            preproc = pipeline.named_steps['preprocessor']
            selector = pipeline.named_steps['feature_selection']
            mask = selector.get_support()
            feature_names = preproc.get_feature_names_out()
            selected_features = feature_names[mask]
            importances = model.feature_importances_
            importance_df = (
                pd.DataFrame({
                    'feature': selected_features,
                    'importance': importances
                })
                .sort_values('importance', ascending=False)
                .reset_index(drop=True)
            )
            print(f"\n------Feature importances using {method}:--------\n")
            print(importance_df)

    def show_study_results(self):
        # ----------------------------------------
        # 4. Results
        # ----------------------------------------
        print("Best f1:", self.__study.best_value)
        print("Best hyperparameters:")
        for k, v in self.__study.best_params.items():
            print(f"{k}: {v}")

    def get_logistic_regression_params(self, trial):
        params = {
            "solver": trial.suggest_categorical("solver", ["lbfgs", "newton-cg", "sag"]),
            "penalty": trial.suggest_categorical("penalty", ["l2", None]),
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "max_iter": trial.suggest_int("max_iter", 100, 2000),
            "l1_ratio": None,  # never used
        }
        return params

    def logistic_regression_hyperparameters_selection(self):
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        def objective(trial):
            params = self.get_logistic_regression_params(trial)
            # oob_score only valid if bootstrap=True
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(**params))
            ])
            # Fit & evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, pos_label=1)
            return f1
        if self.__start_year is not None:
            study_name = f"logistic_regression_{self.__start_year}"
        else:
            study_name = "logistic_regression_full"
        self.__study = optuna.create_study(study_name=study_name, 
                                           direction="maximize",
                                           storage="sqlite:///./optuna_studies/studies_v1.db",
                                           load_if_exists=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.__study.optimize(objective, n_trials=50, show_progress_bar=True) # type: ignore

    def get_decision_tree_params(self, trial):
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 1, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 200),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.01),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        return params

    def decision_tree_hyperparameters_selection(self):
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        def objective(trial):
            params = self.get_decision_tree_params(trial)
            # oob_score only valid if bootstrap=True
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("clf", DecisionTreeClassifier(**params))
            ])
            # Fit & evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            return f1
        if self.__start_year is not None:
            study_name = f"decision_tree_{self.__start_year}"
        else:
            study_name = "decision_tree_full"
        self.__study = optuna.create_study(study_name=study_name, 
                                           direction="maximize",
                                           storage="sqlite:///./optuna_studies/studies_v1.db",
                                           load_if_exists=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.__study.optimize(objective, n_trials=50, show_progress_bar=True) # type: ignore

    def get_gradient_boosting_params(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 1, 8),  # depth of individual trees
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # stochastic boosting
            "validation_fraction": trial.suggest_float("validation_fraction", 0.1, 0.3),
            "n_iter_no_change": trial.suggest_categorical("n_iter_no_change", [None, 5, 10]),
            "tol": trial.suggest_float("tol", 1e-6, 1e-3),
        }
        return params

    def gradient_boosting_hyperparameters_selection(self):
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        def objective(trial):
            params = self.get_gradient_boosting_params(trial)
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("clf", GradientBoostingClassifier(**params))
            ])
            # Fit & evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            return f1
        if self.__start_year is not None:
            study_name = f"gradient_boosting_{self.__start_year}"
        else:
            study_name = "gradient_boosting_full"
        self.__study = optuna.create_study(study_name=study_name, 
                                           direction="maximize",
                                           storage="sqlite:///./optuna_studies/studies_v1.db",
                                           load_if_exists=False)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.__study.optimize(objective, n_trials=50, show_progress_bar=True) # type: ignore

    def train_gradient_boosting(self, params: dict) -> GradientBoostingClassifier:
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        return model

    def get_random_forest_params(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True]),
            "oob_score": trial.suggest_categorical("oob_score", [False, True]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
            "n_jobs": -1,
        }
        return params

    def random_forest_hyperparameters_selection(self):
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        def objective(trial):
            params = self.get_random_forest_params(trial)
            # oob_score only valid if bootstrap=True
            if not params["bootstrap"]:
                params["oob_score"] = False
            # Build pipeline with scaler inside
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("clf", RandomForestClassifier(**params))
            ])
            # Fit & evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            return f1
        if self.__start_year is not None:
            study_name = f"random_forest_{self.__start_year}"
        else:
            study_name = "random_forest_full"
        self.__study = optuna.create_study(study_name=study_name, 
                                           direction="maximize",
                                           storage="sqlite:///./optuna_studies/studies_v1.db",
                                           load_if_exists=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.__study.optimize(objective, n_trials=50, show_progress_bar=True) # type: ignore
    
    def get_xgboost_params(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),  # L1 regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),  # L2 regularization
            "objective": "binary:logistic",
            "tree_method": "hist",  # best for speed + large datasets
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        return params

    def xgboost_hyperparameters_selection(self):
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        def objective(trial):
            params = self.get_xgboost_params(trial)
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("clf", xgb.XGBClassifier(**params, verbosity=0))
            ])
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            return f1
        if self.__start_year is not None:
            study_name = f"xgboost_{self.__start_year}"
        else:
            study_name = "xgboost_full"
        self.__study = optuna.create_study(study_name=study_name,
                                           direction="maximize",
                                           storage="sqlite:///./optuna_studies/studies_v1.db",
                                           load_if_exists=False)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.__study.optimize(objective, n_trials=50, show_progress_bar=True) # type: ignore

    def train_xgboost(self, params: dict) -> xgb.XGBClassifier:
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        model = xgb.XGBClassifier(**params, verbosity=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        return model
    
    def get_dummy_params(self, trial):
        params = {
            "strategy": trial.suggest_categorical("strategy", ["most_frequent", "stratified", "uniform"]),
        }
        return params

    def dummy_hyperparameters_selection(self):
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        def objective(trial):
            params = self.get_dummy_params(trial)
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("clf", DummyClassifier(**params))
            ])
            model.fit(
                X_train, y_train,
            )
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            return f1
        if self.__start_year is not None:
            study_name = f"dummy_{self.__start_year}"
        else:
            study_name = "dummy_full"
        self.__study = optuna.create_study(study_name=study_name,
                                           direction="maximize",
                                           storage="sqlite:///./optuna_studies/studies_v1.db",
                                           load_if_exists=False)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        self.__study.optimize(objective, n_trials=50, show_progress_bar=True) # type: ignore

    def train_dummy_model(self) -> DummyClassifier:
        X_train, y_train, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        return model

    def show_model_results(self, model):
        _, _, X_val, y_val = self.temporal_split(
            target_col="market_path_flag",
            date_col="date",
            test_size_months=self.__test_periods
        )
        y_pred = model.predict(X_val)
        #Show all results including recall, f1-score, accuracy, f1, auc on all labels
        print(classification_report(y_val, y_pred, zero_division=0))

    def temporal_split(self, target_col, date_col, test_size_months=12):
        """
        Chronological train/test split. Returns raw (unscaled) feature DataFrames
        and target Series so scaling can be done inside a pipeline later.
        """
        # order by date
        df = self.__df.sort_values(by=date_col).reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])
        # cutoff: last `test_size_months` months are test
        max_date = df[date_col].max()
        cutoff_date = max_date - pd.DateOffset(months=test_size_months)
        train = df[df[date_col] < cutoff_date].reset_index(drop=True)
        test  = df[df[date_col] >= cutoff_date].reset_index(drop=True)
        # features (DataFrame) and targets (Series) — NO SCALING HERE
        X_train = train.drop(columns=[target_col, date_col])
        y_train = train[target_col].reset_index(drop=True)
        X_test = test.drop(columns=[target_col, date_col])
        y_test = test[target_col].reset_index(drop=True)
        return X_train, y_train, X_test, y_test

    def get_rolling_window_splits(self, date_col, train_months, test_months):
        """
        Genera pares de (Train, Test) moviendo la ventana de entrenamiento hacia adelante.
        La ventana de entrenamiento tiene un tamaño fijo.
        """
        # 1. Preparación de datos (Copia para no afectar el original)
        data = self.__df
        data = data.sort_values(by=date_col)
        splits = []
        # Definimos las fechas límite del dataset
        start_date = data[date_col].min()
        max_date = data[date_col].max()
        # Iteradores
        current_train_start = start_date
        while True:
            # Calcular fechas de corte
            train_end = current_train_start + relativedelta(months=train_months)
            test_end = train_end + relativedelta(months=test_months)
            # Condición de parada: Si el test set se sale del rango de datos
            if test_end > max_date:
                break
            # 2. Filtrado de datos para Train y Test
            # Train: desde inicio actual hasta fin de entrenamiento
            mask_train = (data[date_col] >= current_train_start) & (data[date_col] < train_end)
            # Test: desde fin de entrenamiento hasta fin de prueba
            mask_test = (data[date_col] >= train_end) & (data[date_col] < test_end)
            train_df = data.loc[mask_train]
            test_df = data.loc[mask_test]
            # Guardamos si ambos tienen datos
            # if not train_df.empty and not test_df.empty:
            #     splits.append((train_df, test_df))
            if not train_df.empty and not test_df.empty:
                splits.append(pd.concat([train_df, test_df]))
            # MOVER LA VENTANA (Rolling):
            # El inicio del próximo train se mueve hacia adelante lo que duró el test
            current_train_start = current_train_start + relativedelta(months=test_months)
        return splits
    
    def temporal_split_by_years(
        self,
        train_year_start: int,
        train_year_end: int,
        test_year: int,
        target_col: str,
        date_col: str,
    ):
        """
        Split dataframe into train (two consecutive years) and test (next year).
        Returns X_train, y_train, X_test, y_test.
        """
        df = self.__df.copy()
        # Ensure date is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        # Extract year if not already present
        if "year" not in df.columns:
            df["year"] = df[date_col].dt.year # type: ignore
        # Train years: [start, end]
        train_mask = df["year"].between(train_year_start, train_year_end)
        # Test year
        test_mask = df["year"] == test_year
        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()
        # Split features and labels
        X_train = df_train.drop(columns=[target_col, date_col, "year"])
        y_train = df_train[target_col]
        X_test = df_test.drop(columns=[target_col, date_col, "year"])
        y_test = df_test[target_col]
        return X_train, y_train, X_test, y_test

    def export_best_models_to_mlflow(self):
        storage_url = "sqlite:///./optuna_studies/studies_v1.db"
        # Mapping study_name prefix → model class + param function
        model_registry = {
            "logistic_regression": (LogisticRegression, self.get_logistic_regression_params),
            "random_forest": (RandomForestClassifier, self.get_random_forest_params),
            "xgboost": (xgb.XGBClassifier, self.get_xgboost_params),
            "decision_tree": (DecisionTreeClassifier, self.get_decision_tree_params),
            "gradient_boosting": (GradientBoostingClassifier, self.get_gradient_boosting_params),
            "dummy": (DummyClassifier, self.get_dummy_params),
        }
        Path("./optuna_studies").mkdir(exist_ok=True, parents=True)
        studies = optuna.study.get_all_study_summaries(storage_url)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        for s in studies:
            if not s.study_name.endswith("_full"):
                continue
            print(f"Processing study: {s.study_name}")
            prefix = s.study_name.replace("_full", "")
            if prefix not in model_registry:
                print(f"⚠️ Unknown model prefix: {prefix}, skipping")
                continue
            model_class, get_params_fn = model_registry[prefix]
            # Load Optuna study
            study = optuna.load_study(study_name=s.study_name, storage=storage_url)
            best_trial = study.best_trial
            # Best hyperparameters
            params = get_params_fn(best_trial)
            # Raw (unscaled) train data
            X_train, y_train, _, _ = self.temporal_split(
                target_col="market_path_flag",
                date_col="date",
                test_size_months=self.__test_periods
            )
            # Build pipeline including scaler
            pipeline = Pipeline([
                ("scaler", RobustScaler()),
                ("model", model_class(**params))
            ])
            # Fit full pipeline
            pipeline.fit(X_train, y_train)
            # Log to MLflow as a registered model
            mlflow.set_experiment("best_models")
            with mlflow.start_run(run_name=s.study_name):
                mlflow.log_params(params)
                mlflow.sklearn.log_model( #type: ignore
                    sk_model=pipeline,
                    artifact_path="model",
                    registered_model_name=f"{prefix}_model"
                )
            print(f"✅ Registered: {prefix}_model in MLflow")

    def evaluate_all_models_monthly(self):
        """
        Loads all registered MLflow models and evaluates them month-by-month
        using the raw dataframe (scaling is inside the MLflow pipeline).
        """
        # Load DF sorted by date
        df = self.__df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        # MLflow location
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Get all registered models
        client = mlflow.tracking.MlflowClient() #type: ignore
        registered = client.search_registered_models()
        #Filter out yearly models if any
        registered = [rm for rm in registered if not re.search(r'_(\d{4})$', rm.name)]
        results = []
        # Loop over each saved MLflow model
        for reg_model in registered:
            model_name = reg_model.name
            latest = reg_model.latest_versions
            if not latest:
                continue
            # Take latest version
            mv = latest[0]
            model_uri = f"models:/{model_name}/{mv.version}"
            print(f"Evaluating model: {model_name} (v{mv.version})")
            model = mlflow.sklearn.load_model(model_uri) #type: ignore
            # Loop year-month combinations
            df["year"] = df["date"].dt.year #type: ignore
            df["month"] = df["date"].dt.month #type: ignore
            year_months = df[["year", "month"]].drop_duplicates()
            for _, row in year_months.iterrows():
                y = row["year"]
                m = row["month"]
                df_month = df[(df["year"] == y) & (df["month"] == m)]
                if df_month.empty:
                    continue
                X = df_month.drop(columns=["market_path_flag", "date", "year", "month"])
                y_true = df_month["market_path_flag"]
                # Predict using full pipeline (scaler + model)
                y_pred = model.predict(X) #type: ignore
                # Compute metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[0, 1], average=None, zero_division=0
                )
                # Overall
                precision_overall = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", pos_label=1, zero_division=0
                )[0]
                recall_overall = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", pos_label=1, zero_division=0
                )[1]
                f1_overall = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", pos_label=1, zero_division=0
                )[2]
                # Calculate ROC AUC
                try:
                    y_proba = model.predict_proba(X)[:, 1] #type: ignore
                    roc_auc = roc_auc_score(y_true, y_proba)
                except Exception as e:
                    roc_auc = 0
                # Save row
                results.append({
                    "model_name": model_name,
                    "year": y,
                    "month": m,
                    "precision_overall": precision_overall,
                    "recall_overall": recall_overall,
                    "f1_overall": f1_overall,
                    "roc_auc": roc_auc,
                    # Class 0 metrics
                    "precision_0": precision[0], #type: ignore
                    "recall_0": recall[0], #type: ignore
                    "f1_0": f1[0], #type: ignore
                    # Class 1 metrics
                    "precision_1": precision[1], #type: ignore
                    "recall_1": recall[1], #type: ignore
                    "f1_1": f1[1], #type: ignore
                })
        # Return dataframe
        metrics_df = pd.DataFrame(results)
        metrics_df = metrics_df.sort_values(["model_name", "year", "month"]).reset_index(drop=True)
        return metrics_df

    def plot_metric_over_time(self, df, metric_col):
        model_col = "model_name" if "model_name" in df.columns else "model"
        df = df.copy()
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" +
                                    df["month"].astype(str) + "-01")
        df = df.sort_values("date")
        # --- 1. LÓGICA DE RESALTADO ESTATICO ---
        HIGHLIGHT_MODELS = {"random_forest_model", "dummy_model", "random_forest", "dummy"}
        # Definir colores fijos para los modelos a resaltar
        STATIC_COLORS = {
            "random_forest_model": 'red',     # Color llamativo 1
            "dummy_model": 'darkgreen',       # Color llamativo 2
            "random_forest": 'red',           # Color llamativo 1
            "dummy": 'darkgreen',             # Color llamativo 2
        }
        # Paleta de colores suaves para el resto
        unique_models = df[model_col].unique()
        # Contar cuántos modelos NO son los resaltados para asignarles un color único
        other_models = [m for m in unique_models if m not in HIGHLIGHT_MODELS]
        NUM_OTHER_MODELS = len(other_models)
        # Usamos 'Set2' o 'Pastel1' para tonos suaves.
        # Usamos cm.get_cmap(..., extend='max') solo si fuera necesario, aquí no lo es.
        cmap = cm.get_cmap('Set2', max(1, NUM_OTHER_MODELS))
        # 2. Mapeo de color para todos los modelos
        model_colors = {}
        other_models_counter = 0
        for model in unique_models:
            if model in HIGHLIGHT_MODELS:
                # Asignar el color estático y asegurar un fallback
                model_colors[model] = STATIC_COLORS.get(model, 'black')
            else:
                # Asignar un color de la paleta suave
                model_colors[model] = cmap(other_models_counter)
                other_models_counter += 1
        plt.figure(figsize=(12, 6))
        # 3. Dibujar las líneas
        for model in unique_models:
            sub = df[df[model_col] == model]
            color = model_colors[model]
            # Ajustar grosor y zorder
            if model in HIGHLIGHT_MODELS:
                lw = 4  # Más grueso para los modelos resaltados
                zorder = 10 # Asegura que la línea esté al frente
            else:
                lw = 1.5
                zorder = 1
            plt.plot(
                sub["date"],
                sub[metric_col],
                marker="o",
                label=model,
                linewidth=lw,
                color=color,
                zorder=zorder 
            )
        # El resto del código
        last_date = df["date"].max()
        test_start = last_date - pd.DateOffset(months=12)
        plt.axvline(test_start, color="black", linewidth=2)
        plt.xlabel("Date")
        plt.ylabel(metric_col)
        plt.title(f"{metric_col} over time by model (Random Forest and Dummy Highlighted)")
        # --- CAMBIO CLAVE: Posicionar la leyenda fuera y abajo a la izquierda ---
        plt.legend(
            loc='upper left',        # Punto de anclaje de la leyenda (esquina superior izquierda de la caja de la leyenda)
            bbox_to_anchor=(0, -0.1), # Coordenadas: (0, 0) es la esquina inferior izquierda del gráfico.
            ncol=2,                  # Opcional: Para poner las entradas en 2 columnas para ahorrar espacio horizontal
            fancybox=True,
            shadow=True
        )
        # ------------------------------------------------------------------------
        plt.grid(True)
        plt.show()

    def show_best_model_params_mlflow(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        client = MlflowClient()
        # New API for listing models
        registered_models = client.search_registered_models()
        if not registered_models:
            print("No registered models found in MLflow.")
            return
        for rm in registered_models:
            model_name = rm.name
            print(f"\n=== Model: {model_name} ===")
            # Each model has versions; pick the latest version
            latest_version = rm.latest_versions[-1]  # type: ignore
            run_id = latest_version.run_id
            # Fetch run data
            run = client.get_run(run_id)
            params = run.data.params
            print("Best Parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")

    def export_yearly_models_to_mlflow(self):
        """
        Export all best models from yearly Optuna studies (xxx_YYYY) to MLflow,
        keeping them well-organized inside each registered model by tagging
        each MLflow model version with training/testing years.
        """
        storage_url = "sqlite:///./optuna_studies/studies_v1.db"
        # Model mapping
        model_registry = {
            "logistic_regression": (LogisticRegression, self.get_logistic_regression_params),
            "random_forest": (RandomForestClassifier, self.get_random_forest_params),
            "xgboost": (xgb.XGBClassifier, self.get_xgboost_params),
            "decision_tree": (DecisionTreeClassifier, self.get_decision_tree_params),
            "gradient_boosting": (GradientBoostingClassifier, self.get_gradient_boosting_params),
            "dummy": (DummyClassifier, self.get_dummy_params),
        }
        # Ensure Optuna folder exists
        Path("./optuna_studies").mkdir(exist_ok=True, parents=True)
        # Load all study summaries
        studies = optuna.study.get_all_study_summaries(storage_url)
        # MLflow settings
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("best_models_by_year")
        client = MlflowClient()
        # Regex to detect yearly studies: prefix_YYYY
        pattern = re.compile(r"(.+?)_(\d{4})$")
        for s in studies:
            match = pattern.match(s.study_name)
            if not match:
                continue  # skip non-yearly studies
            prefix, year_str = match.groups()
            year = int(year_str)
            if prefix not in model_registry:
                print(f"⚠️ Unknown prefix '{prefix}', skipping study {s.study_name}")
                continue
            print(f"📌 Processing study by year: {s.study_name}")
            model_class, get_params_fn = model_registry[prefix]
            # Load Optuna study
            study = optuna.load_study(study_name=s.study_name, storage=storage_url)
            best_trial = study.best_trial
            # Get hyperparams
            params = get_params_fn(best_trial)
            # Compute training and test periods
            train_year_start = year
            train_year_end = year + 1   # training is (year, year+1)
            test_year = year + 2
            # Get raw data
            X_train, y_train, _, _ = self.temporal_split_by_years(
                train_year_start=train_year_start,
                train_year_end=train_year_end,
                test_year=test_year,
                target_col="market_path_flag",
                date_col="date"
            )
            # Build pipeline
            pipeline = Pipeline([
                ("scaler", RobustScaler()),
                ("model", model_class(**params))
            ])
            # Fit the model
            pipeline.fit(X_train, y_train)
            # Register in MLflow
            run_name = f"{prefix}_train_{train_year_start}_{train_year_end}_test_{test_year}"
            with mlflow.start_run(run_name=run_name) as run:
                # Log params
                mlflow.log_params(params)
                # Log yearly metadata
                mlflow.set_tags({
                    "train_year_start": train_year_start,
                    "train_year_end": train_year_end,
                    "test_year": test_year,
                    "model_prefix": prefix,
                    "study_name": s.study_name,
                })
                # Log model as a new version of prefix_model
                mlflow.sklearn.log_model( #type: ignore
                    sk_model=pipeline,
                    artifact_path="model",
                    registered_model_name=f"{prefix}_model_{train_year_start}"
                )
            print(f"   ✔️ Saved model {prefix}_model for {train_year_start}-{train_year_end} → {test_year}")

    def evaluate_yearly_models_monthly(self):
        """
        Evaluates YEARLY MLflow models where model names follow:
            <prefix>_model_<year>

        Meaning:
        - model_year = x
        - trained on x and x+1
        - evaluated on x+2 (full year, month-by-month)
        """
        # Prepare raw dataframe
        df = self.__df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year # type: ignore
        df["month"] = df["date"].dt.month # type: ignore
        df = df.sort_values("date")
        # MLflow connection
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        client = mlflow.tracking.MlflowClient() #type: ignore
        # YEARLY MODEL PATTERN: prefix_model_YYYY
        pattern = re.compile(r"^(.*)_model_(\d{4})$")
        results = []
        registered = client.search_registered_models()
        for reg in registered:
            model_name = reg.name
            # Match prefix_model_year
            m = pattern.match(model_name)
            if not m:
                continue
            model_prefix = m.group(1)
            train_year_start = int(m.group(2))       # x
            train_year_end = train_year_start + 1    # x+1
            test_year = train_year_start + 2         # x+2
            # Load latest version
            latest_versions = reg.latest_versions
            if not latest_versions:
                continue
            mv = latest_versions[0]
            model_uri = f"models:/{model_name}/{mv.version}"
            print(f"Evaluating {model_name} → test_year={test_year}")
            # Load pipeline
            model = mlflow.sklearn.load_model(model_uri) #type: ignore
            # Filter target test year
            df_test_year = df[df["year"] == test_year]
            if df_test_year.empty:
                continue
            # List all months available in test year
            year_months = (
                df_test_year[["year", "month"]]
                .drop_duplicates()
                .sort_values(["year", "month"])
            )
            for _, row in year_months.iterrows():
                y = row["year"]
                m = row["month"]
                df_month = df_test_year[df_test_year["month"] == m]
                if df_month.empty:
                    continue
                X = df_month.drop(columns=["market_path_flag", "date", "year", "month"])
                y_true = df_month["market_path_flag"]
                # Predict using MLflow pipeline
                y_pred = model.predict(X) # type: ignore
                # Class-specific metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[0, 1], average=None, zero_division=0
                )
                # Overall metrics (binary: pos_label=1)
                precision_o, recall_o, f1_o, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", pos_label=1, zero_division=0
                )
                # ROC AUC
                try:
                    y_proba = model.predict_proba(X)[:, 1] # type: ignore
                    roc_auc = roc_auc_score(y_true, y_proba)
                except Exception as e:
                    roc_auc = 0
                results.append({
                    "model_name": model_prefix,
                    "train_year_start": train_year_start,
                    "train_year_end": train_year_end,
                    "test_year": test_year,
                    "year": y,
                    "month": m,
                    "precision_overall": precision_o,
                    "recall_overall": recall_o,
                    "f1_overall": f1_o,
                    "precision_0": precision[0], # type: ignore
                    "recall_0": recall[0], # type: ignore
                    "f1_0": f1[0], # type: ignore
                    "precision_1": precision[1], # type: ignore
                    "recall_1": recall[1], # type: ignore
                    "f1_1": f1[1], # type: ignore
                    "roc_auc": roc_auc,
                })
        metrics = pd.DataFrame(results)
        metrics = metrics.sort_values(["model_name", "year", "month"]).reset_index(drop=True)
        return metrics

