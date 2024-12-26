# Source https://github.com/robertvacareanu/llm4regression/blob/37b98e60170d5b68399915440b72dd9bd88b702e/src/regressors/sklearn_regressors.py

import random
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.kernel_ridge import KernelRidge


def linear_regression(x_train, x_test, y_train, y_test, random_state=1):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'linear_regression',
        'x_train'  : x_train,
        'x_test'   : x_test,
        'y_train'  : y_train,
        'y_test'   : y_test,
        'y_predict': y_predict,
    }


def ridge(x_train, x_test, y_train, y_test, random_state=1):
    model = Ridge(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'ridge',
        'x_train'  : x_train,
        'x_test'   : x_test,
        'y_train'  : y_train,
        'y_test'   : y_test,
        'y_predict': y_predict,
    }

def lasso(x_train, x_test, y_train, y_test, random_state=1):
    model = Lasso(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'lasso',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

# See `Minimum Width for Universal Approximation` (says it's max(input_dim+1, output_dim))

def mlp_universal_approximation_theorem1(x_train, x_test, y_train, y_test, random_state=1):
    """
    Wide MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_uat_1',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def mlp_universal_approximation_theorem2(x_train, x_test, y_train, y_test, random_state=1):
    """
    Wide MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_uat_2',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


def mlp_universal_approximation_theorem3(x_train, x_test, y_train, y_test, random_state=1):
    """
    Wide MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(1000, ), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_uat_3',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def mlp_deep1(x_train, x_test, y_train, y_test, random_state=1):
    """
    Deep MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_deep1',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def mlp_deep2(x_train, x_test, y_train, y_test, random_state=1):
    """
    Deep MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(10, 20, 10), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_deep2',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def mlp_deep3(x_train, x_test, y_train, y_test, random_state=1):
    """
    Deep MLP
    """
    model = MLPRegressor(hidden_layer_sizes=(10, 20, 30, 20, 10), activation='relu', solver='lbfgs', random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'mlp_deep3',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def random_forest(x_train, x_test, y_train, y_test, random_state=1):
    """
    Random Forest Regressor
    """
    model = RandomForestRegressor(max_depth=3, random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'random_forest',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def bagging(x_train, x_test, y_train, y_test, random_state=1):
    """
    Bagging Regressor
    """
    model = BaggingRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'bagging',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def gradient_boosting(x_train, x_test, y_train, y_test, random_state=1):
    """
    Gradient Boosting Regressor
    """
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'gradient_boosting',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


def adaboost(x_train, x_test, y_train, y_test, random_state=1):
    """
    AdaBoost Regressor
    """
    model = AdaBoostRegressor(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'adaboost',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


def voting(x_train, x_test, y_train, y_test, random_state=1):
    """
    Voting Regressor
    """
    model = VotingRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'voting',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }


# def stacking(x_train, x_test, y_train, y_test, random_state=1):
#     """
#     Stacking Regressor
#     """
#     model = StackingRegressor()
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     y_test    = y_test.to_numpy()

#     return {
#         'model_name': 'stacking',
#         'x_train'   : x_train,
#         'x_test'    : x_test,
#         'y_train'   : y_train,
#         'y_test'    : y_test,
#         'y_predict' : y_predict,
#     }

def bayesian_regression1(x_train, x_test, y_train, y_test, random_state=1):
    model = make_pipeline(
        PolynomialFeatures(degree=10, include_bias=False),
        StandardScaler(),
        BayesianRidge(),
    )

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'bayesian_regression',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def svm_regression(x_train, x_test, y_train, y_test, random_state=1):
    model = SVR()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'svm',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def svm_and_scaler_regression(x_train, x_test, y_train, y_test, random_state=1):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), SVR())
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'svm_w_s',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression(x_train, x_test, y_train, y_test, random_state=1):
    model = KNeighborsRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_v2(x_train, x_test, y_train, y_test, random_state=1):
    model = KNeighborsRegressor(weights='distance')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn_v2',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_v3(x_train, x_test, y_train, y_test, random_state=1):
    model = KNeighborsRegressor(n_neighbors=3, weights='distance')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn_v3',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_v4(x_train, x_test, y_train, y_test, random_state=1):
    model = KNeighborsRegressor(n_neighbors=1, weights='distance')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn_v4',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_v5_adaptable(x_train, x_test, y_train, y_test, random_state=1):
    """
    The idea behind this function is to have a KNN model that adapts to the size of the training set
    Presumably, when you have very little training data, you want to use a small number of neighbors
    As the number of examples increase, a larger numbers of neighbors is fine.
    """
    if x_train.shape[0] < 3:
        n_neighbors=1
    elif x_train.shape[0] < 7:
        n_neighbors=3
    else:
        n_neighbors=5

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'knn_v5_adaptable',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_generic(x_train, x_test, y_train, y_test, model_name, knn_kwargs):
    # knn_args = {**knn_kwargs}
    # if 'n_neighbors' in knn_args:
    #     knn_args['n_neighbors'] = min(knn_args['n_neighbors'], len(x_train))
    model = KNeighborsRegressor(**knn_kwargs)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': model_name,
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def knn_regression_search():
    """
    A large number of KNN variants
    """
    idx = 0
    knn_fns = []
    for n_neighbors in [1, 2, 3, 5, 7, 9, 11]:
        for weights in ['uniform', 'distance']:
            for p in [0.25, 0.5, 1, 1.5, 2]:
                idx += 1
                knn_fns.append(
                    lambda x_train, x_test, y_train, y_test: knn_regression_generic(x_train, x_test, y_train, y_test, model_name=f'knn_search_{idx}', knn_kwargs={'n_neighbors': n_neighbors, 'weights': weights, 'p': p})
                )
    return knn_fns

def kernel_ridge_regression(x_train, x_test, y_train, y_test, random_state=1):
    model = KernelRidge()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'kernel_ridge',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def lr_with_polynomial_features_regression(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    degree = kwargs.get('degree', 2)
    
    # Create a pipeline that first transforms the data using PolynomialFeatures, then applies Linear Regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'lr_with_polynomial_features',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def spline_regression(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    n_knots = kwargs.get('degree', 5) # Same defaults as SplineTransformer
    degree  = kwargs.get('degree', 3) # Same defaults as SplineTransformer
    
    # Create a pipeline that first transforms the data using PolynomialFeatures, then applies Linear Regression
    model = Pipeline([
        ('spline', SplineTransformer(n_knots=n_knots, degree=degree)),
        ('linear', LinearRegression())
    ])
    
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_test    = y_test.to_numpy()

    return {
        'model_name': 'spline',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def baseline_average(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    pred = np.mean(y_train)
    y_predict = np.array([pred for _ in y_test])
    y_test = y_test.to_numpy()

    return {
        'model_name': 'average',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def baseline_last(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    pred = y_train.to_list()[-1]
    y_predict = np.array([pred for _ in y_test])
    y_test = y_test.to_numpy()

    return {
        'model_name': 'last',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def baseline_random(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    r = random.Random(random_state)
    y_train_list = y_train.to_list()
    y_predict = np.array([r.choice(y_train_list) for _ in y_test])
    y_test = y_test.to_numpy()

    return {
        'model_name': 'random',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def baseline_constant(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    y_predict = np.array([kwargs['constant_prediction_value'] for _ in y_test])
    y_test = y_test.to_numpy()

    return {
        'model_name': 'constant_prediction',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def linear_regression_manual_gd(x_train, x_test, y_train, y_test, random_state=1, **kwargs):
    """
    Linear regression using manual gradient descent steps.
    
    Parameters:
    - steps: number of gradient descent steps (default: 2)
    - learning_rate: step size for gradient descent (default: 0.01)
    """
    steps = kwargs.get('steps', 2)
    learning_rate = kwargs.get('learning_rate', 0.01)
    
    # Convert to numpy arrays if they aren't already
    x_train = x_train.to_numpy() if hasattr(x_train, 'to_numpy') else np.array(x_train)
    y_train = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.array(y_train)
    x_test = x_test.to_numpy() if hasattr(x_test, 'to_numpy') else np.array(x_test)
    y_test = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.array(y_test)
    
    # Initialize parameters (weights and bias)
    n_features = x_train.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    
    # Perform gradient descent steps
    m = len(x_train)
    for _ in range(steps):
        # Forward pass
        y_pred = np.dot(x_train, weights) + bias
        
        # Compute gradients
        dw = (1/m) * np.dot(x_train.T, (y_pred - y_train))
        db = (1/m) * np.sum(y_pred - y_train)
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
    
    # Make predictions on test set
    y_predict = np.dot(x_test, weights) + bias

    return {
        'model_name': f'linear_regression_gd_{steps}_steps',
        'x_train'   : x_train,
        'x_test'    : x_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'y_predict' : y_predict,
    }

def create_linear_regression_gd_variants(
    steps_options=[1, 2, 3, 4],
    learning_rates=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
    init_weights_options=['zeros', 'ones', 'random', 'random_uniform'],
    momentum_values=[0.0, 0.5, 0.9],  # 0.0 means no momentum
    lr_schedules=['constant', 'linear_decay', 'exponential_decay']
):
    """
    Factory function that creates multiple variants of linear regression with gradient descent,
    each with different hyperparameters.
    
    Args:
        steps_options (list): Number of gradient descent steps to try
        learning_rates (list): Learning rates to try
        init_weights_options (list): Weight initialization strategies to try
        momentum_values (list): Momentum values to try (0.0 means no momentum)
        lr_schedules (list): Learning rate schedule strategies to try
    
    Returns:
        List of functions, each implementing linear regression with specific hyperparameters
    """
    variants = []
    
    # Create all combinations of hyperparameters
    configs = []
    for steps in steps_options:
        for lr in learning_rates:
            for init in init_weights_options:
                for momentum in momentum_values:
                    for lr_schedule in lr_schedules:
                        configs.append({
                            'steps': steps,
                            'learning_rate': lr,
                            'init_weights': init,
                            'momentum': momentum,
                            'lr_schedule': lr_schedule
                        })
    
    def create_gd_function(steps, learning_rate, init_weights, momentum, lr_schedule):
        def linear_regression_gd(x_train, x_test, y_train, y_test, random_state=1):
            # Convert to numpy arrays
            x_train = x_train.to_numpy() if hasattr(x_train, 'to_numpy') else np.array(x_train)
            y_train = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.array(y_train)
            x_test = x_test.to_numpy() if hasattr(x_test, 'to_numpy') else np.array(x_test)
            y_test = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.array(y_test)
            
            # Initialize parameters
            n_features = x_train.shape[1]
            np.random.seed(random_state)
            
            if init_weights == 'zeros':
                weights = np.zeros(n_features)
            elif init_weights == 'ones':
                weights = np.ones(n_features)
            elif init_weights == 'random':
                weights = np.random.randn(n_features) * 0.01
            elif init_weights == 'random_uniform':
                weights = np.random.uniform(-0.01, 0.01, n_features)
            
            bias = 0
            
            # Initialize momentum vectors if using momentum
            if momentum > 0:
                v_weights = np.zeros_like(weights)
                v_bias = 0
            
            # Perform gradient descent steps
            m = len(x_train)
            for step in range(steps):
                # Compute effective learning rate based on schedule
                if lr_schedule == 'constant':
                    current_lr = learning_rate
                elif lr_schedule == 'linear_decay':
                    current_lr = learning_rate * (1 - step/steps)
                elif lr_schedule == 'exponential_decay':
                    current_lr = learning_rate * (0.95 ** step)
                
                # Forward pass
                y_pred = np.dot(x_train, weights) + bias
                
                # Compute gradients
                dw = (1/m) * np.dot(x_train.T, (y_pred - y_train))
                db = (1/m) * np.sum(y_pred - y_train)
                
                # Apply momentum if using it
                if momentum > 0:
                    v_weights = momentum * v_weights - current_lr * dw
                    v_bias = momentum * v_bias - current_lr * db
                    weights += v_weights
                    bias += v_bias
                else:
                    weights = weights - current_lr * dw
                    bias = bias - current_lr * db
            
            # Make predictions
            y_predict = np.dot(x_test, weights) + bias

            return {
                'model_name': f'lr_gd_s{steps}_lr{learning_rate}_i{init_weights}_m{momentum}_sc{lr_schedule}',
                'x_train'   : x_train,
                'x_test'    : x_test,
                'y_train'   : y_train,
                'y_test'    : y_test,
                'y_predict' : y_predict,
            }
        
        return linear_regression_gd
    
    # Create a function for each configuration
    for config in configs:
        variants.append(
            create_gd_function(
                steps=config['steps'],
                learning_rate=config['learning_rate'],
                init_weights=config['init_weights'],
                momentum=config['momentum'],
                lr_schedule=config['lr_schedule']
            )
        )
    
    return variants