#Import modules for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
import csv, spacy, string, re, pandas,nltk
from nltk import ngrams
from collections import Counter
from PyDictionary import PyDictionary
dictionary = PyDictionary()


class Dataset(object):

    def __init__(self, language):
        self.language = language
        #loading spaCy language model
        if language == 'english':
            self.nlp = spacy.load('en_core_web_sm')
        else:
            self.nlp = spacy.load('es_core_news_sm')

        #loading training and testing set
        trainset_path = "datasets/{}/{}_Train.tsv".format(language, language.capitalize())
        testset_path = "datasets/{}/{}_Test.tsv".format(language, language.capitalize())

        self.trainset = self.read_dataset(trainset_path)
        self.testset = self.read_dataset(testset_path)

        uni = []
        for sent in self.trainset:
            uni += (sent['sentence'].split(" "))
        #Compute unigram count through out training instance
        self.unigram = Counter(uni)
        #make the whole training set as a spaCy object
        doc_train = self.nlp(''.join(uni))
        char_tri = []
        for word in uni:
            string_word = re.sub('[^A-Za-z]+', '', word)
            char_3 = ngrams(string_word.lower(), 3)
            for tri in char_3:
                char_tri.append(tri)
        #Compute character trigram throughout the whole training instance
        char_trig = list(Counter(char_tri).keys())
        self.char_trigram = list(map(''.join, char_trig))

        suf = []
        pos = []
        dep = []
        Shape = []
        #spaCy library operation for pos tag, suffix tag, syntactic dependencies
        #tags and shape tags of a word instance or a word phrase
        for token in doc_train:
            suf.append(token.suffix_)
            pos.append(token.pos_)
            dep.append(token.dep_)
            Shape.append(token.shape_)

        #Frequency table for both English and Spanish
        if language == 'english':
            letter_frequency = {'a':0.08167, 'b':0.01492, 'c':0.02782, 'd':0.04253, 'e':0.12702,'f':0.02228,'g':0.02015,
                                 'h':0.06094,'i':0.06966,'j':0.00153,'k':0.00772,'l':0.04025,'m':0.02406,'n':0.06749,
                                 'o':0.07507,'p':0.01929,'q':0.00095,'r':0.05987,'s':0.06327,'t':0.09056,'u':0.02758,
                                 'v':0.00978,'w':0.02360,'x':0.00150,'y':0.01974,'z':0.00074 }
            self.frequency = letter_frequency
        else:
            letter_frequency = {'a':0.11525,'b':0.02215,'c':0.04019,'d':0.05010,'e':0.12181,'f':0.00692,'g':0.01768,
                                 'h':0.00703,'i':0.06247,'j':0.00493,'k':0.00011,'l':0.04967,'m':0.03157,'n':0.06712,
                                 'o':0.08682,'p':0.02510,'q':0.00877,'r':0.06871,'s':0.07977,'t':0.04632,'u':0.02927,
                                 'v':0.01138,'w':0.00017,'x':0.00215,'y':0.01008,'z':0.00467,'á':0.00502,'é':0.0433,
                                 'í':0.00725,'ñ':0.00311,'ó':0.00827,'ú':0.00168,'ü':0.00012}
            self.frequency = letter_frequency

        #Compute suffix count throughout the whole training set
        self.suffix = Counter(suf)
        pos_tag = []
        detail_t = []
        for tag in pos:
            pos_tag.append(tag.lower())
        for detail_tag in tag:
            detail_t.append(detail_tag.lower())

        #Compute part of speech tag as a list of all possibel pos tags
        self.pos = list(Counter(pos_tag).keys())
        #Computer the count for syntactic dependencies and the shape of word as a dictionary object
        self.dep = Counter(dep)
        self.shape = Counter(Shape)

    def read_dataset(self, file_path):
        with open(file_path) as file:
            fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

            dataset = [sent for sent in reader]

        return dataset

#reference(frequency table):
#https://en.wikipedia.org/wiki/Letter_frequency
