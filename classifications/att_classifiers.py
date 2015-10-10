"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from helper_classes import Feature_Preprocessor

def SVM_predict_rank(features, classes, unknown, actual_classes):
    """
        Proviced a ranking of the different authors by likelyhood of having authored each unknown text.
    """
    FP = Feature_Preprocessor(features, True, False, 30)
    features = FP.batch_normalize(features)
    unknown = FP.batch_normalize(unknown)

    clf = SVC(probability=True, kernel='rbf', C=2.4, degree=1, gamma=0.7/len(features[0]))
    clf.fit(features, classes)

    # I'm sorry about the following lines:
    predictions = map(lambda x : zip(clf.classes_, x), clf.predict_log_proba(unknown))
    orderings = zip(map(lambda x : sorted(x, key = lambda s : s[1], reverse=True), predictions), actual_classes)

    orderings = [([ e[0] for e in l[0] ], l[1]) for l in orderings]
    rankings = map(lambda x : x[0].index(x[1]), orderings )

    return rankings

def SVM_predict(features, classes, unknown):
    """
        Provices the most likely author for each unknown text
    """
    FP = Feature_Preprocessor(features, True, False, 30)
    features = FP.batch_normalize(features)
    unknown = FP.batch_normalize(unknown)

    clf = SVC(kernel='rbf', C=2.4, degree=1, gamma=0.7/len(features[0]))
    clf.fit(features, classes)

    return clf.predict(unknown)

def KNeighborsClassifier_predict(features, classes, unknown):
    """
        Provices the most likely author for each unknown text
    """
    FP = Feature_Preprocessor(features, True, False, 30)
    features = FP.batch_normalize(features)
    unknown = FP.batch_normalize(unknown)

    clf = KNeighborsClassifier(n_neighbors=4,  weights='distance', algorithm='brute', metric='minkowski', p=1)
    clf.fit(features, classes)

    return clf.predict(unknown)

def DecisionTreeClassifier_predict(features, classes, unknown):
    """
        Provices the most likely author for each unknown text
    """
    FP = Feature_Preprocessor(features, True, True, 30)
    features = FP.batch_normalize(features)
    unknown = FP.batch_normalize(unknown)

    clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, splitter='best')
    clf.fit(features, classes)

    return clf.predict(unknown)

def AdaBoostClassifier_predict(features, classes, unknown):
    """
        Provices the most likely author for each unknown text
    """
    FP = Feature_Preprocessor(features, True, False, 30)
    features = FP.batch_normalize(features)
    unknown = FP.batch_normalize(unknown)

    clf = AdaBoostClassifier(n_estimators=10, learning_rate=0.998, algorithm='SAMME.R', random_state=1)
    clf.fit(features, classes)

    return clf.predict(unknown)

