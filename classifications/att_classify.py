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

from random import sample, seed, shuffle

from att_classifiers import *


def name_has_substring(name, substrings):
    """
        Checks if a name contains one of many substrings
    """
    for elem in substrings:
        if elem in name:
            return True
    return False


def create_splits_attack(data, num_authors=40, samples=1, exclude=['verification', 'imitation'], attack=['obfuscation']):
    """
       Splits the data into learn-test sets.
       Seed is set to have identical results on different runs.
    """
    sets = []
    seed(1)
    for authorset in [ sample(sorted(data.keys()), num_authors) for _ in xrange(samples) ]:
        inset_f = []
        inset_c = []
        outset_f = []
        outset_c = []
        for author in authorset:
            for story in data[author].keys():
                if not name_has_substring(story, exclude):
                    if name_has_substring(story, attack):
                        outset_f.append(data[author][story])
                        outset_c.append(author)
                    else:
                        inset_f.append(data[author][story])
                        inset_c.append(author)
        sets.append(((inset_f, inset_c),(outset_f, outset_c)))

    return sets


def create_splits(data, samples=2, num_authors=40, splits_per_sample=1, exclude=['verification', 'imitation', 'obfuscation']):
    """
        Splits the data into learn-test sets.
        A seed is set for randomization, and items are sorted, so as to give same results on different runs and on different systems
    """
    seed(1)

    # Get a clean data set of only natural writing styles
    clean_stories = dict()
    for author in data.keys():
        author_stories = filter(lambda x : not name_has_substring(x, exclude), data[author].keys())
        clean_stories[author] = dict([ (k, data[author][k]) for k in author_stories])

    assert min([len(x) for x in clean_stories.values()]) >= splits_per_sample, "More splits per sample than there are samples"

    # A data dictionary with the stories for each author sorted so that we have same order (and same results) on different operating systems
    author_to_storylist = dict( [(a, sorted(clean_stories[a].keys())) for a in clean_stories.keys()] )
    sets = []

    for subset in [ sample(sorted(clean_stories.keys()), num_authors) for _ in xrange(samples) ]:
        subset = dict([ (k, clean_stories[k]) for k in subset ])
        [ shuffle(author_to_storylist[k]) for k in author_to_storylist.keys() ]
        for index in xrange(splits_per_sample):
            inset_f = []
            inset_c = []
            outset_f = []
            outset_c = []
            for author in subset:
                for i in xrange(len(author_to_storylist[author])):
                    if i == index:
                        outset_f.append(subset[author][author_to_storylist[author][i]])
                        outset_c.append(author)
                    else:
                        inset_f.append(subset[author][author_to_storylist[author][i]])
                        inset_c.append(author)

            sets.append(((inset_f, inset_c),(outset_f, outset_c)))

    return sets


def cross_validate(sets, method, verbose=True):
    """
        Performs a cross validation using _method_ on _sets_
        It just returns the results of the prediction method (with factual results), whatever it might be that the prediction method returns
    """
    from time import time
    def get_score(method, (inset_f, inset_c),(outset_f, outset_c)):
        prediction = method(inset_f, inset_c, outset_f)
        return prediction, outset_c

    start = time()
    p, c = [], []
    for s in sets:
        pr, cl = get_score(method, *s)
        [ p.append(x) for x in pr ]
        [ c.append(x) for x in cl ]
    stop = time()
    if verbose:
        print "\tTime spent:", int((stop - start)/60.0)
    return p, c


def data_select_specific_features(data, features=False):
    """
        Selects specific features from a data set and returns data set in similar structure.
    """
    from itertools import chain
    flatten = lambda x : list(chain(*x))

    newdata = dict()
    for name in data.keys():
        author = dict()
        for key, (f, d) in data[name].items():
            if features == False:
                author[key] = f
            else:
                author[key] = flatten( [ d[n] for n in features ] )
        newdata[name] = author
    return newdata


def get_precision_at_rank(sets, method=SVM_predict_rank):
    """
        Returns the precision at all ranks for sets
    """
    print "Determining precision at rank."

    def precisions_at_ranks(ranks, setsize):
        precisions = []
        for i in xrange(setsize):
            precisions.append( sum([ r <= i for r in ranks ])/float(len(ranks)) )
        return precisions

    def cross_validate(sets, method):
        def get_score(method, (inset_f, inset_c),(outset_f, outset_c)):
            return method(inset_f, inset_c, outset_f, outset_c)

        rankings = []
        for s in sets:
            rankings += get_score(method, *s)
        return rankings

    rankings = cross_validate(sets, method)
    precisions = precisions_at_ranks(rankings, len(set(sets[0][0][1])))
    print precisions


if __name__ == '__main__':
    print "Loading data.."
    from feature_extraction.Cached_Features import data
    print "Working..."

    features = ['mono_char_dist', 'mono_chunk_dist', 'bi_tag_dist', 'word_length', 'legomena', 'bi_char_dist', 'readability', 'mono_tag_dist']

    selected_data = data_select_specific_features(data, features)
    attack = False
    if attack:
        sets = create_splits_attack(selected_data, num_authors=10, samples=2)
    else:
        sets = create_splits(selected_data, samples=10, num_authors=10, splits_per_sample=2)

    print get_precision_at_rank(sets)