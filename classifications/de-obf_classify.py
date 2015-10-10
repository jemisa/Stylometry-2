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

from random import seed, sample

from helper_classes import Feature_Preprocessor

from att_classifiers import SVM_predict_rank
from att_classify import name_has_substring, data_select_specific_features


def AdaBoostClassifier_predict_texttype(features, classes, unknown):
    """
        Predicts the type of a text (binary classification)
        Parameters optimized for natural vs obfuscated.
    """
    from sklearn.ensemble import AdaBoostClassifier

    FP = Feature_Preprocessor(features, True, False, 30)
    features = FP.batch_normalize(features)
    unknown = FP.batch_normalize(unknown)

    clf = AdaBoostClassifier(n_estimators=80, learning_rate=0.998, algorithm='SAMME.R', random_state=1)
    clf.fit(features, classes)

    return clf.predict(unknown)


def create_splits(data, samples=10, exclude=['verification', 'imitation', 'obfuscation'], attack=['obfuscation']):
    """
        Creates splits of 39 authors to learn natural vs obfuscation
            and then the author of an obfuscated text must be identified on the basis of only natural texts.
        Works similarly as att_classify.create_splits_attack, but also ads obf vs natural pairs.
    """
    sets = []
    seed(1)     # Set seed to always have same outcome
    authorsets = [ sample(sorted(data.keys()), 40) for _ in xrange(samples) ]   # sort for same outcome on different systems
    for authorset in authorsets:    # Loop over different selections of authors
        for exclude_author in authorset:    # Loop over different authors to leave out and attribute the obfuscated text of.
            inset_f = []
            inset_c = []
            outset_f = []
            outset_c = []
            reg_obf_pairs = []

            for story in data[exclude_author].keys():
                if name_has_substring(story, attack):
                    outset_f.append(data[exclude_author][story])
                    outset_c.append(exclude_author)
                else:
                    if not name_has_substring(story, exclude):
                        inset_f.append(data[exclude_author][story])
                        inset_c.append(exclude_author)

            for include_author in authorset:
                if not include_author == exclude_author:
                    regular_texts = []
                    obfuscation = None  # There only ever is one of those in the EBG data set
                    for story in data[include_author].keys():
                        if name_has_substring(story, attack):
                            obfuscation = data[include_author][story]
                        else:
                            if not name_has_substring(story, exclude):
                                regular_texts.append(data[include_author][story])
                                inset_f.append(data[include_author][story])
                                inset_c.append(include_author)
                    reg_obf_pairs.append((regular_texts, obfuscation))

            sets.append(((inset_f, inset_c),(outset_f, outset_c), reg_obf_pairs))

    return sets


def get_precision_at_rank(sets, deobf='detect', method=SVM_predict_rank):
    """
        Similar to att_classify.get_precision_at_rank
        Adds the posibility of de-obfuscation.
            deobfuscation can be done: ['never', 'detect', 'always']
            So, never deobfuscate, only if obfuscation is detected, or always.
    """
    def precisions_at_ranks(ranks, setsize):
        """
            Calculate the recall/precision at each rank.
        """
        precisions = []
        for i in xrange(setsize):
            precisions.append( sum([ r <= i for r in ranks ])/float(len(ranks)) )
        return precisions

    def cross_validate(sets, deobf, method):
        def get_score(method, (inset_f, inset_c), (outset_f, outset_c), pairs):
            if (deobf == 'detect' and text_is_obfuscated(outset_f, pairs)) or deobf == 'always':
                outset_f = perform_deobfuscation(outset_f[0], pairs)
            return method(inset_f, inset_c, outset_f, outset_c)

        rankings = []
        for s in sets:
            rankings += get_score(method, *s)
        return rankings

    rankings = cross_validate(sets, deobf, method)
    precisions = precisions_at_ranks(rankings, len(set(sets[0][0][1])))
    return precisions


def perform_deobfuscation(outset_f, pairs):
    """
        Perform deobfuscation of _outset_f_ by learning the obfuscationbehavior of _pairs_
    """
    def learn_obf_behavior(pairs):
        difference_v = lambda a, b : [ x[0] - x[1] for x in zip(a,b) ]
        def average_v(vectors, lr=1):
            # Calculate the average vector of a list of vectors.
            total = [ 0.0 for _ in xrange(len(vectors[0])) ]
            for vec in vectors:
                total = map(lambda x : x[0] + x[1], zip(total, vec))
            vec_count = float(len(vectors))
            return [ lr * (v / vec_count) for v in total ]

        difference_vectors = []
        for (reg, obf) in pairs:
            difference_vectors.append( difference_v(average_v(reg), obf) )

        return average_v(difference_vectors)

    return [[ x[0] + x[1] for x in zip(learn_obf_behavior(pairs), outset_f) ]]


def text_is_obfuscated(outset_f, pairs):
    """
        Decide based on _pairs_ if text described as _outset_f_ is obfuscated or not
    """
    regular = []
    obfuscated = []
    for (reg, obf) in pairs:
        obfuscated.append(obf)
        for r in reg:
            regular.append(r)
    features = regular + obfuscated
    classes = [ 0 for _ in xrange(len(regular)) ] + [ 1 for _ in xrange(len(obfuscated)) ]
    return AdaBoostClassifier_predict_texttype(features, classes, outset_f)[0] == 1


if __name__ == '__main__':
    print "Loading data.."
    from feature_extraction.Cached_Features import data
    print "Working..."

    features=['mono_char_dist', 'mono_chunk_dist', 'bi_tag_dist', 'word_length', 'legomena', 'bi_char_dist', 'readability', 'mono_tag_dist']

    samples = 1 # Set to high number for more accurate mearusements
    data = data_select_specific_features(data, features)
    sets = create_splits(data, samples=samples)

    average = lambda x : sum(x) / len(x)

    for deobf in ['never', 'detect', 'always']:
        ranks = get_precision_at_rank(sets, deobf=deobf)
        print "deobf:"+deobf, ",  ave(recall):", average(ranks), ranks

    print "Done"