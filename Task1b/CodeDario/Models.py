from sklearn.neural_network import MLPRegressor as MLP
from sklearn.linear_model import Ridge as RDG
from sklearn.linear_model import LinearRegression as LR


class Regression():
    def LinRegression(X_data, y_data):
        model =LR(fit_intercept=True,
                normalize=False,
                copy_X=True,
                n_jobs=1)
        model.fit(X_data, y_data)

        weights = model.coef_
        return weights

    def RidgeRegression(X_data, y_data):
        model =RDG(alpha=0.01,
                copy_X=True,
                fit_intercept=True,
                max_iter=None,
                solver='svd',
                tol=0.001)
        model.fit(X_data, y_data)
        
        weights = model.coef_
        return weights

    def PerceptronRegression(X_data, y_data):
        model = MLP(hidden_layer_sizes=(5,5),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.001,
                    tol=0.000001,
                    max_iter=20000,
                    shuffle=True,
                    verbose=0)
        model.fit(X_data, y_data)

        weights = model.coef_
        return weights