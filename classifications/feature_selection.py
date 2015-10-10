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

from att_classify import cross_validate, create_splits, create_splits_attack, data_select_specific_features

from att_classifiers import *


def feature_names(data):
    """
        Get the names of the different features that are available in _data_
    """
    return data[data.keys()[0]].values()[0][1].keys()


def get_precision(prediction, classes):
    """
        Returns the relative number of correct answers.
    """
    return sum([ 1 if x[0]==x[1] else 0 for x in zip(prediction, classes)])/float(len(classes))


def creat_good_featureset_BU(data, method, attack=False, selection=[], heavy=False):
    """
        Selects the best features Bottum Up:
        Start with precision of -1, then each time select the feature that yields the largest improvement
    """
    print "Creating good featureset Bottum Up"
    def find_best_addition(data, method, selection, left, current_precision, attack):
        best_name = False
        for name in left:
            precision = get_precision_for_configuration(data, method, features=selection + [name], attack=attack)
            if precision > current_precision:
                current_precision = precision
                best_name = name
        return best_name, current_precision

    names = feature_names(data)

    left = names
    for elem in selection:
        left.remove(elem)
    current_precision = -1
    if len(selection) > 0:
        current_precision = get_precision_for_configuration(data, method, features=selection, attack=attack, heavy=heavy)
    print "Startprecision:", current_precision
    while True:
        addition, current_precision = find_best_addition(data, method, selection, left, current_precision, attack)
        if addition == False:
            break
        else:
            print "\tFound improvement:", addition, current_precision
            selection.append(addition)
            left.remove(addition)
    return current_precision, selection


def creat_good_featureset_TD(data, method, selection=False, attack=False, heavy=False):
    """
        Selects the best features Top Down:
        Start with precision of [using all features], then each time eliminate the feature that yields the largest improvement
    """
    print "Creating good featureset Top Down"
    def find_best_removal(data, method, selection, current_precision, attack):
        best_name = False
        for name in selection:
            precision = get_precision_for_configuration(data, method, features=set(selection) - set([name]), attack=attack, heavy=heavy)
            if precision > current_precision:
                current_precision = precision
                best_name = name
        return best_name, current_precision

    if selection == False:
        selection = feature_names(data)

    current_precision = get_precision_for_configuration(data, method, selection, attack)
    print "Startprecision:", current_precision
    while True:
        removal, current_precision = find_best_removal(data, method, selection, current_precision, attack)
        if removal == False:
            break
        else:
            print "\tFound improvement:", removal, current_precision
            selection.remove(removal)
    return current_precision, selection


def rank_features_solo(data, method, heavy=False):
    """
        Calculate and rank how well features perform on an individual basis
    """
    print "Ranking features solo"
    names = feature_names(data)

    pairs = []
    for name in names:
        precision = get_precision_for_configuration(data, method, features=[name], heavy=heavy)
        pairs.append((precision, name))
    ranking = sorted(pairs, key=lambda x : x[0], reverse=True)
    for elem in ranking:
        print elem
    print


def rank_features_dropout(data, method, heavy=False):
    """
        Calculate and rank how much noise individual features introduce i.c.w. the whole.
    """
    print "Ranking features based on dropout..."
    selected_data = data_select_specific_features(data, False)
    sets = create_splits(selected_data)
    base_precision = get_precision(*cross_validate(sets, method=method, verbose=False))
    print "Base precision:", base_precision

    names = feature_names(data)

    pairs = []
    for name in names:
        precision = get_precision_for_configuration(data, method, heavy=heavy, features=set(names)-set([name]))
        pairs.append((precision-base_precision, name))
    ranking = sorted(pairs, key=lambda x : x[0])
    for elem in ranking:
        print elem
    print


def survey(data, attack=False, heavy=False, features=False):
    """
        Survey common machine learning methods on the data
        In heavy mode: performs large cross validation.
        Otherwise: performs very light testing.
    """
    data = data_select_specific_features(data, features)

    if not heavy:
        samples = 2
        splits_per_sample = 1
    else:
        samples = 80
        splits_per_sample = 13

    if attack:
        sets = create_splits_attack(data, samples)
    else:
        sets = create_splits(data, samples, splits_per_sample=splits_per_sample)

    methods = [
        ('NearestNeighbors', KNeighborsClassifier_predict),
        ('svm', SVM_predict),
        #('AdaBoost', AdaBoostClassifier_predict),
        #('DecisionTreeClassifier', DecisionTreeClassifier_predict),
        ]

    texttype = "natural"
    if attack:
        texttype = "attack"
    msg = "Performing a survey for " + texttype + " texts ("
    if not heavy:
        msg += "non-"
    print msg + "heavy)"
    print "(uniform) prior:", 1.0/len(sets[0][1][1])

    for (name, method) in methods:
        print "Cross validating", name, "..."
        print "   \t", get_precision(*cross_validate(sets, method=method, verbose=False))
    print


def get_precision_for_configuration(data, method, features=False, heavy=False, attack=False):
    """
        Get the precision for data, method, heavy, features, and attack parameters.
    """
    selected_data = data_select_specific_features(data, features)
    if not heavy:
        samples = 2
        splits_per_sample = 1
    else:
        samples = 80
        splits_per_sample = 13

    if attack:
        sets = create_splits_attack(selected_data, samples=samples)
    else:
        sets = create_splits(selected_data, samples=samples, splits_per_sample=splits_per_sample)

    return get_precision(*cross_validate(sets, method=method, verbose=False))


if __name__ == '__main__':
    print "Loading data.."
    from feature_extraction.Cached_Features import data
    print "Working..."

    # Use _heavy_ = False for a short demo/test
    # Use _heavy_ = True for accurate results
    heavy = False

    survey(data, attack=False, heavy=heavy)

    survey(data, attack=True, heavy=heavy)

    rank_features_solo(data, SVM_predict, heavy=heavy)
    rank_features_dropout(data, SVM_predict, heavy=heavy)

    print creat_good_featureset_TD(data, SVM_predict, attack=False, heavy=heavy)
    print
    print creat_good_featureset_BU(data, SVM_predict, attack=False, heavy=heavy)