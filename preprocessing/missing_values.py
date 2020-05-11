from sklearn.base import TransformerMixin
import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


class MissingValuesInserter(TransformerMixin):
    def __init__(self, columns, percentage=0.01, nan_representation=None):
        self.columns = columns
        self.percentage = percentage
        self.nan_representation = nan_representation

    def fit(self):
        return self

    def fit_transform(self, X, *_):
        """
        :param X: a numpy array or pandas DataFrame
        :param _:
        :return: a numpy array or pandas DataFrame with imputed null values
        """
        is_numpy_array = isinstance(X, np.ndarray)
        is_data_frame = isinstance(X, pd.DataFrame)
        if not is_data_frame and not is_numpy_array:
            raise ValueError('X is not DataFrame or numpy array!')
        data_size = X.shape[0]
        no_objects_to_imputs_missing_values = round(self.percentage * data_size)
        indexes_to_put_missing_values = np.full(shape=(1, no_objects_to_imputs_missing_values), fill_value=-1)
        for i in range(0, no_objects_to_imputs_missing_values):
            random_index_candidate = random.randint(0, data_size - 1)
            while random_index_candidate in indexes_to_put_missing_values:
                random_index_candidate = random.randint(0, data_size - 1)
            indexes_to_put_missing_values.put(i, random_index_candidate)
        if is_numpy_array:
            X_copy = np.copy(X)
            ixgrid = np.ix_(indexes_to_put_missing_values[0], self.columns)
            X_copy[ixgrid] = self.nan_representation
            return X_copy

        X_copy = X.copy()
        X_copy.loc[indexes_to_put_missing_values[0], self.columns] = self.nan_representation
        return X_copy


class MissingValuesInserterColumnsIndependent(TransformerMixin):
    def __init__(self, columns, percentage=0.01, nan_representation=None):
        self.columns = columns
        self.percentage = percentage
        self.nan_representation = nan_representation

    def fit(self):
        return self

    def fit_transform(self, X, *_):
        X_prim = X
        for i in range(0, len(self.columns)):
            if i == 0:
                X_prim = MissingValuesInserter(columns=(self.columns[i],), percentage=self.percentage, nan_representation=self.nan_representation).fit_transform(X)
            else:
                X_prim = MissingValuesInserter(columns=(self.columns[i],), percentage=self.percentage, nan_representation=self.nan_representation).fit_transform(X_prim)
        return X_prim


if __name__ == '__main__':
    # working with in-memory DataFrame
    bmi = pd.DataFrame.from_dict({'height': [1.6, 1.6, 1.7, 1.8], 'weight': [60, 80, 80, 85], 'bmi': [1, 0, 0, 1]})
    print(bmi)
    msImptr = MissingValuesInserter(columns=('height', 'weight'), percentage=0.5)
    print('missing values imputed to columns:', ('height', 'weight'), ' with percentage = ', 0.5)
    missing_values_imputed = msImptr.fit_transform(bmi)
    print(missing_values_imputed)
    print(missing_values_imputed.to_numpy())
    missing_values_imputed.to_csv('./dummy.csv')
    print('missing values imputed to columns ', ('height', 'weight'), ' independently with percentage = ', 0.5)
    independent = MissingValuesInserterColumnsIndependent(columns=('height', 'weight'), percentage=0.5).fit_transform(bmi)
    print(independent)
    filled = independent
    filled = independent.fillna(1000000000)
    print(filled)
    k = KNeighborsClassifier(n_neighbors=2, algorithm='auto').fit(filled.loc[:, ('height', 'weight')], filled.loc[:, 'bmi'])
    print(k.predict([[1.6, 75]]))

    # working with numpy arrays
    X,y = load_iris(return_X_y=True)
    transformer = MissingValuesInserterColumnsIndependent(columns=[0, 1], percentage=0.5)
    missing = transformer.fit_transform(X)





