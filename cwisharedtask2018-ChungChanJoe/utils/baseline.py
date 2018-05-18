#Baseline
import spacy,re, nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk import ngrams

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        # load spaCy language model for englisha and spanish
        if language == 'english':
            self.avg_word_length = 5.3
            self.nlp = spacy.load('en_core_web_sm')

        else:  # spanish
            self.avg_word_length = 6.2
            self.nlp = spacy.load('es_core_news_sm')

#Implementing Random Forest Classification Model for the system and fix the random state seed.
        self.model = RandomForestClassifier(random_state = 170122089)

#Extract features from a word or word phrase instance
    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        uni_count = 0
        frequency_total = 0
        number_of_letter = 0
        string_word = re.sub('[^A-Za-z]+', '', word)
        for character in string_word:
            frequency_total += self.frequency[character.lower()]
            number_of_letter += 1
        if number_of_letter != 0:
            frequency_average = frequency_total/number_of_letter
        else:
            frequency_average = frequency_total
        num_words = 0
        trig = []
        for uni in word.split(' '):
            char_t = ngrams(uni,3)
            for tri in char_t:
                trig.append(tri)
            uni_count += self.unigram[uni]

        wd = self.nlp(word)
        suffix_count = 0
        pos_tag_seq= []
        dep_count = 0
        shape_count = 0
        token_count = 0

        #Getting the recognition using spaCy
        for token in wd:
            suffix_count += self.suffix[token.suffix_]
            pos_tag_seq.append(token.pos_)
            dep_count += self.dep[token.dep_]
            shape_count += self.shape[token.shape_]
            token_count += 1

        #get part of speech as a feature of vectors with One Hot Encoder
        pos_vec = CountVectorizer(vocabulary = self.pos)
        pos_tag_seq = [','.join(pos_tag_seq)]
        p = pos_vec.fit_transform(pos_tag_seq)
        pos_count = p.toarray()[0]

        #get character as the same form as part of speech
        trigram = list(map(''.join, trig))
        char_tri_vec = CountVectorizer(vocabulary = self.char_trigram)
        char_tri = [','.join(trigram)]
        character_trigram = char_tri_vec.fit_transform(char_tri)
        char_tri_count = character_trigram.toarray()[0]

        y = [len_chars, len_tokens, suffix_count, uni_count, shape_count, frequency_average, dep_count]
        #Do .extend to merge the two vector towards the vector with all other single features
        y.extend(pos_count)
        y.extend(char_tri_count)


        return y

    def train(self, trainset,uni,suf,char_tri, pos_tag, dep, shape, frequency):
        self.unigram = uni
        self.suffix = suf
        self.char_trigram = char_tri
        self.pos = pos_tag
        self.dep = dep
        self.shape = shape
        self.frequency = frequency

        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])
        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)
