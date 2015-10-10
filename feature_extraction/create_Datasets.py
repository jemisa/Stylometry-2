# -*- coding: utf-8 -*-

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

import shelve, os
from itertools import chain
flatten = lambda x : list(chain(*x))

from pattern.en import parsetree
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag, map_tag
simplify_tag = lambda t : map_tag('en-ptb', 'universal', t)


def process_raw_text(text):
    """
        First some code to standardize the formatting, then basic nlp.
    """
    # Remove breaks and tabs
    for char in ["\t", "\n"]:
        text = text.replace(char, " ")
    text = text.replace('."', '".')
    text = text.replace(".'", "'.")
    # Split special characters from words
    for char in ["'", '"', ",", ".", "?", "!", ";", ":"]:
        text = text.replace(char, " " + char + " ")
    # Magic to remove all multi-spaces
    text = ' '.join(text.split())

    # get the words, sentences, POS tags, and chunks.
    chunks = [ tuple([ c.type for c in t.chunks ]) for t in parsetree(text) ]
    sentences = sent_tokenize(text)
    sentences = [ word_tokenize(s) for s in sentences ]
    sentences_tags = [ tuple([ (w, simplify_tag(t)) for w, t in pos_tag(s) ]) for s in sentences ]

    sentences = [ tuple([ w for w, _ in s]) for s in sentences_tags ]
    tags = [ tuple([ t for _, t in s]) for s in sentences_tags ]
    words = flatten(sentences)

    return tuple(words), tuple(sentences), tuple(tags), tuple(chunks)


def load_file(filename='text.txt'):
    """
        Reads all text in filename, returns the following triplet:
        - list of all words
        - sentences (ordered list of words, per sentence)
        - POS-tags (ordered list of tags, per sentence)
        - chunks
    """
    f = open(filename,'r')
    text = "".join([ x + " " for x in f.readlines() ]).decode("utf8")
    f.close()
    return process_raw_text(text)


def process_blog(filename):
    """
        This reads in a bloggers argive, and splits up the posts
        It is filled with early returns that return None
            in case certain criteria are not met.
    """
    def name_to_info(name):
        info = name.split('/').pop().split('.')
        info.pop()
        blog_id = info.pop(0)
        return blog_id, tuple(info)

    if 'indUnk' in filename:
        return (None, None), None

    f = open(filename, 'r')
    content = "".join([ x + " " for x in f.readlines() ])
    f.close()

    content = content.split("<post>")
    content.pop(0)
    if not len(content) > 14:
        return (None, None), None

    stories = [ x.split('</post>')[0].decode("utf8", 'ignore') for x in content ]
    stories = filter(lambda x : len(x) > (510 * 3), stories)

    if not len(content) > 14:
        return (None, None), None

    stories = map(process_raw_text, stories)
    stories = filter(lambda x : len(x[0]) > 510, stories)
    if not len(stories) > 14:
        return (None, None), None
    else:
        return name_to_info(filename), stories


def create_cached_dataset_blogs(datafolder, cachelocation="../rawb/"):
    """
        Create the blog data set.
    """
    blogs = sorted(filter(lambda x : not x.startswith("."), os.listdir(datafolder)))
    cache = os.path.join(datafolder, cachelocation)
    for i in xrange(len(blogs)):
        b = blogs[i]
        if i % 100 == 0:
            print "\tWorking on:", i, '\t', (datafolder + b).split("/").pop()
        (blog_id, info), posts = process_blog(datafolder + b)
        if not blog_id == None:
            sh = shelve.open(cache + 'blog_' + blog_id + '.shelve')
            sh[blog_id] = (info, posts)
            sh.close()
    print "Done!"


def create_cached_dataset(datafolder):
    """
        Creates data set for the Drexel AMT corpus.
    """
    folders = filter(lambda x : not x.startswith("."), os.listdir(datafolder))

    dataset = dict()
    for folder in folders:
        print "Working on:", folder
        dataset[folder] = dict()
        files = filter(lambda x : not x.startswith("."), os.listdir(datafolder + folder))
        for f in files:
            if 'demographics' not in f:
                w, s, t, c = load_file(datafolder + folder + "/" + f)
                dataset[folder][f] = (w, s, t, c)
    f = open(os.path.dirname(os.path.realpath(__file__)) + "Dataset.py", 'w')
    f.write("# -*- coding: utf-8 -*-\n")
    f.write("data = " + str(dataset) + "\n")
    f.close()


def demo(datafolder):
    # Performs action just on one text file
    folder = filter(lambda x : not x.startswith("."), os.listdir(datafolder)).pop()
    f = filter(lambda x : not x.startswith("."), os.listdir(datafolder + folder)).pop()
    print "Working on:", folder, f

    w, s, t, c = load_file(datafolder + folder + "/" + f)
    print w
    print s
    print t
    print c
    print type(w), type(w[0]), len(w)
    print type(s), type(s[0]), len(s)
    print type(t), type(t[0]), len(t)
    print type(c), type(c[0]), len(c)


if __name__ == '__main__':
    from os import path

    #datafolder = path.join(path.dirname(path.realpath(__file__)), "../Data/Drexel-AMT-Corpus/")
    #demo(datafolder)
    #create_cached_dataset(datafolder)

    datafolder = path.join(path.dirname(path.realpath(__file__)), "../../blogs/")
    create_cached_dataset_blogs(datafolder)

