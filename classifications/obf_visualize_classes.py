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

from helper_classes import Feature_Preprocessor
from att_classify import data_select_specific_features, name_has_substring

def get_feature_vectors_from_data(data, interest=['obfuscation'], exclude=['verification','imitation']):
    """
       Get the features and classes from the data
    """
    set_f = []
    set_c = []
    for in_name in data.keys():
        for n, f in data[in_name].items():
            if not name_has_substring(n, exclude):
                set_f.append(f)
                if name_has_substring(n, interest):
                    set_c.append(1)
                else:
                    set_c.append(0)

    return set_f, set_c

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    print "Loading data.."
    from feature_extraction.Cached_Features import data
    print "Normalizing..."

    # Select features
    data = data_select_specific_features(data, ['bi_char_dist', 'legomena', 'word_length', 'tri_char_dist', 'mono_tag_dist', 'sentence_length', 'readability'])

    # Get the data separated in features and classes
    features, classes = get_feature_vectors_from_data(data)

    # Compres the features to two numbers (points)
    FP = Feature_Preprocessor(features, False, True, 2)
    features = FP.batch_normalize(features)


    print "Data processed, now plotting..."

    # Convert a list of points to two lists of x and y points (fortran style)
    x = [ p[0] for p in features ]
    y = [ p[1] for p in features ]

    # Split into points of interest and normal points (obfuscated texts ad natural texts)
    interest = [ p for p in zip(x,y,classes) if p[2] == 1 ]
    normal = [ p for p in zip(x,y,classes) if p[2] == 0 ]

    # scatterplot the points
    norml = plt.scatter(zip(*normal)[0], zip(*normal)[1], marker='x', c='b', s=40)
    inter = plt.scatter(zip(*interest)[0], zip(*interest)[1], marker='o', c='r', s=40)

    # Some metaplots and show
    plt.ylabel('Principal Component #2', fontsize=24)
    plt.xlabel('Principal Component #1', fontsize=24)

    plt.legend((norml, inter),
               ('Natural', 'Obfuscated'),
               scatterpoints=1,
               loc='lower right',
               ncol=1,
               fontsize=28)

    plt.show()

