import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

df = pd.read_csv('StudentPerformanceFactors.csv')

#Categorical and numerical columns are described for further use
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64']).columns

for col in categorical_cols:
    #NaN values are replaced with the mode value for each categorical column
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numerical_cols:
    #NaN values are replaced with the mean value for each numerical column
    df[col] = df[col].fillna(df[col].mean())

#Categorical values are converted into numercial ones for further analyze
df = pd.get_dummies(df, columns=categorical_cols)

#Numerical values are set to lower and upper bounds decided by interquartile range
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

#Dataframe is split into input and output columns
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

#Dataframe is split into train and test parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Data is standardized to fit alrogithms
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mse_scores = []
r2_scores = []

#Parameters and algorithms for GridSearch are defined
param_grids = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },
    "Ridge Regression": {
        "model": Ridge(),
        "params": {
            "alpha": [0.1, 1.0, 10.0],
            "solver": ["auto", "sparse_cg", "saga"]
        }
    },
    "Lasso Regression": {
        "model": Lasso(),
        "params": {
            "alpha": [0.1, 1.0, 10.0],
            "max_iter": [1000, 5000, 10000],
            "tol": [1e-4, 1e-3]
        }
    },
    "KNN Regressor": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    },
    "Decision Tree Regressor": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5]
        }
    }
}

scorer = make_scorer(mean_squared_error, greater_is_better=False)

#Parameters are sent into GridSearch for cross validation
for model_name, config in param_grids.items():
    grid_search = GridSearchCV(estimator=config["model"],
    param_grid=config["params"],
    scoring=scorer,
    cv=5,
    n_jobs=1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    #Best models and its parameters are printed on screen
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name}:")
    print(f"Best Parameters: {best_params}")
    print(f"CV MSE: {best_score:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R²: {r2:.4f}")

    #The model is saved when alrogithm has an average of R² value greater than 0.9
    if r2 > 0.9:
        model_filename = f"{model_name.replace(' ', '_')}_best_model.pkl"
        joblib.dump(best_model, model_filename)
        print(f"{model_name} model has been saved as {model_filename}")

    print("----------------------")