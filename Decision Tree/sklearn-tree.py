from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import tree
import numpy as np
import pandas as pd

if __name__ == '__main__':
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    # print(lenses_target)

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)

    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)

    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)

    clf = tree.DecisionTreeClassifier(max_depth = 4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    print('The prediction for [1, 1, 1, 0] is:')
    print(clf.predict([[1, 1, 1, 0]]))