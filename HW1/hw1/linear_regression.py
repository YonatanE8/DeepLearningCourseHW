import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.dot(X,   self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.

        w_opt = None
        # ====== YOUR CODE: ======
        # The optimal solution for the linear regression optimization problem (using the L2 metric) is given by:
        # W* = (lambda * I + X^T * X)^-1 * X^T * Y - (Psuedo-Inverse)
        w_opt = np.dot(np.dot(np.linalg.inv((self.reg_lambda * np.eye(X.shape[1]) + np.dot(X.T, X))), X.T), y)
        # ========================

        self.weights_ = w_opt
        
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)

        # TODO: Add bias term to X as the first feature.

        xb = None
        # ====== YOUR CODE: ======
        shape = X.shape
        num_of_dims = len(shape)
        if num_of_dims == 1:
            xb = np.concatenate((np.array([1]), X))
            
        else:
            xb = np.concatenate((np.ones([shape[0], 1]), X), axis=1)
            
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """
    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        pass
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        # check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # TODO: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        transformer = PolynomialFeatures(self.degree)
        X_poly = transformer.fit_transform(X)
        X_log = np.expand_dims(np.log(X[:, 0]), axis=1)
        X_log = np.concatenate((X_log, np.expand_dims(np.log(X[:, -1]), axis=1)), axis=1)
        X_transformed = np.concatenate((X_poly, X_log), axis = 1)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    # Load into a pandas dataframe and show some samples
    names = df.columns
    
    # Extract the statistics of MEDV - To be used with all other features
    medv = df[names[-1]].get_values()
    medv_mean = np.mean(medv)
    medv_std = np.std(medv)
    norm_medv = medv - medv_mean
    norm_medv_sqrt = np.sqrt(np.sum(np.power(norm_medv, 2)))

    corrs = []
    for i, name in enumerate(names):
        if i == len(names) - 1:
            break
            
        # Compute the current feature statistics
        feature = df[name].get_values()
        feature_mean = np.mean(feature)
        feature_std = np.std(feature)
        norm_feature = feature - feature_mean

        # Compute the correlation constant
        norm_feature_sqrt = np.sqrt(np.sum(np.power(norm_feature, 2)))
        std_prod = norm_medv_sqrt * norm_feature_sqrt
        corr_coeff = np.sum((norm_feature * norm_medv)) / std_prod
        
        corrs.append(corr_coeff)
    
    corrs = np.array(corrs)
    sorted_inds = np.argsort(np.abs(corrs))
    top_features_inds = sorted_inds[-n:]
    top_features_inds = np.array([top_features_inds[(-i - 1)] for i in range(len(top_features_inds))])
    top_n_features = np.array(names)[top_features_inds].tolist()
    top_n_corr = corrs[top_features_inds].tolist()
    # ========================

    return top_n_features, top_n_corr


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    samples = np.arange(X.shape[0])
    fold_size = X.shape[0] // k_folds
    
    inds_start = [(k * fold_size) for k in range(k_folds)]
    
    mean_scores = []
    for i, degree in enumerate(degree_range):
        feature_transformer = BostonFeaturesTransformer(degree=degree)
            
        for j, lambda_ in enumerate(lambda_range):
            regressor = LinearRegressor(reg_lambda=lambda_)
    
            scores = []
            for k in range(k_folds):
                if k == 0:
                    X_train = X[inds_start[1]:, :]
                    X_test = X[0:inds_start[1]]
                    
                    y_train = y[inds_start[1]:]
                    y_test = y[0:inds_start[1]]
                
                elif k == k_folds - 1:
                    X_train = X[0:inds_start[-1], :]
                    X_test = X[inds_start[-1]:]
                    
                    y_train = y[0:inds_start[-1]]
                    y_test = y[inds_start[-1]:]
                    
                else:
                    X_test = X[inds_start[k]:(inds_start[k] + fold_size), :]
                    y_test = y[inds_start[k]:(inds_start[k] + fold_size)]
                    
                    X_train_1 = X[0:inds_start[k], :]
                    X_train_2 = X[(inds_start[k] + fold_size):, :]
                    X_train = np.concatenate((X_train_1, X_train_2), axis=0)
                    
                    y_train_1 = y[0:inds_start[k]]
                    y_train_2 = y[(inds_start[k] + fold_size):]
                    y_train = np.concatenate((y_train_1, y_train_2), axis=0)
                            
                    transformed_features_train = feature_transformer.transform(X_train)
                    regressor.fit(transformed_features_train, y_train)
                                        
                    transformed_features_test = feature_transformer.transform(X_test)
                    y_pred = regressor.predict(transformed_features_test)
                
                    mse = np.mean(np.power((y_test - y_pred), 2))
                    scores.append(mse)
                    
            mean_scores.append([np.mean(scores), i, j])
            
    best_score = 10 ** 9
    for l in mean_scores:
        sc, i, j = l
        
        if sc < best_score:
            degree = degree_range[i]
            lambda_ = lambda_range[j]
            best_score = sc
    
   
    best_params = {'bostonfeaturestransformer__degree': degree, 'linearregressor__reg_lambda': lambda_}
    # ========================

    return best_params
