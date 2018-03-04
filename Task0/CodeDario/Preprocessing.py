from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


def minmax(X_data):
    scaler = MinMaxScaler(feature_range=(0,1))
    X_data_centered = scaler.fit_transform(X_data)
    return X_data_centered