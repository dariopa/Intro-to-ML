from sklearn.linear_model import Ridge as RDG

def RidgeRegression(alpha, X_data, y_data, X_test):
        model =RDG(alpha=alpha,
        copy_X=True,
        fit_intercept=True,
        max_iter=20000,
        solver='cholesky')
        model.fit(X_data, y_data)
        
        return model.predict(X_test)
