from sklearn.neural_network import MLPRegressor as MLP
from sklearn.linear_model import Ridge as RDG
from sklearn.linear_model import LinearRegression as LR


class Regression():
    def LinRegression(X_data, y_data, X_test):
        model =LR(fit_intercept=True,
                solver='auto',
                normalize=False,
                copy_X=True,
                n_jobs=1)
        model.fit(X_data, y_data)

        y = model.predict(X_test)
        return y

    def RidgeRegression(X_data, y_data, X_test):
        model =RDG(alpha=0,
                copy_X=True,
                fit_intercept=True,
                max_iter=None,
                tol=0.00000000000001)
        model.fit(X_data, y_data)

        y = model.predict(X_test)
        return y

    def PerceptronRegression(X_data, y_data, X_test):
        model = MLP(hidden_layer_sizes=(5,5),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.001,
                    tol=0.000001,
                    max_iter=20000,
                    shuffle=True,
                    verbose=0)
        model.fit(X_data, y_data)

        y = model.predict(X_test)
        return y