from statistics import stdev
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from classification.decomposition import OneVsRestClassifierForRandomBinaryClassifier
from dataset.aggregations import *
from preprocessing.missing_values import MissingValuesInserterColumnsIndependent
from classification.k_neighbours import KNNAlgorithmF, KNNAlgorithmM
import setup_path as pth

vertebral = pd.read_csv(pth.concatenate_path_os_independent('column_3C.dat'), sep=' ')
X = vertebral.iloc[:, :-1].to_numpy()
y = vertebral.iloc[:, -1].to_numpy()
y = LabelEncoder().fit_transform(y)

X = MinMaxScaler().fit_transform(X)

missing = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
dfs = []
ind = 0
ks = [(2, 4), (3, 5)]
for miss in missing:
    for agg in (A1Aggregation(), A2Aggregation(), A3Aggregation(), A4Aggregation(p=3), A5Aggregation(), A6Aggregation(),
                A7Aggregation(), A8Aggregation(), A9Aggregation(), A10Aggregation()):
        for r in (2, 5, 10):
            for k in ks:
                X_missing = MissingValuesInserterColumnsIndependent(columns=range(X.shape[1]), nan_representation=-1, percentage=miss)\
                    .fit_transform(X)
                multiclassF = OneVsRestClassifierForRandomBinaryClassifier(KNNAlgorithmF(missing_representation=-1, r=r, aggregation=agg, k_neighbours=k))
                f_result = cross_validate(multiclassF, X_missing, y, scoring='roc_auc_ovo', return_estimator=True, cv=10)
                w = pd.DataFrame({'algorithm': 'f', 'k': str(k), 'r': f_result['estimator'][0].estimator.r,
                              'agg': Aggregation.change_aggregation_to_name(f_result['estimator'][0].estimator.aggregation), 'missing': miss,
                                  'auc': np.mean(f_result['test_score']), 'stddev': stdev(f_result['test_score'])}, index=[ind])
                print(w)
                dfs.append(w)
                ind += 1
                concatenated = pd.concat(dfs)

concatenated = pd.concat(dfs)
concatenated.to_excel('concatenated_vertebral.xlsx')

for miss in missing:
    for k in ks:
        X_missing = MissingValuesInserterColumnsIndependent(columns=range(X.shape[1]), nan_representation=-1, percentage=miss) \
            .fit_transform(X)
        multiclassM = OneVsRestClassifier(KNNAlgorithmM(missing_representation=-1, k_neighbours=k))
        m_result = cross_validate(multiclassM, X_missing, y, scoring='roc_auc_ovo', return_estimator=True, cv=10)
        w2 = pd.DataFrame({'algorithm': 'm', 'k': str(k), 'r': '',
                           'agg': '',
                           'missing': miss,
                           'auc': np.mean(m_result['test_score']), 'stddev': stdev(m_result['test_score'])},
                          index=[ind])
        print(w2)
        dfs.append(w2)
        ind += 1
concatenated = pd.concat(dfs)
concatenated.to_excel('concatenated_vertebral_m.xlsx')
