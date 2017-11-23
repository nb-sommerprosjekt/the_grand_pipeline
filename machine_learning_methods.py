# Inspired by https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/py3_poc.py
import gensim
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import os
import dill
import pickle

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        #self.dim = len(list(word2vec)[0])
        self.dim = 300
        #self.dim = len(list(word2vec.values())[0])
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    #or [np.zeros(300)], axis = 0)
                    or [np.zeros(self.dim)], axis = 0 )
            for words in X

        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        #self.dim = len(list(word2vec)[0])
        self.dim = 300
        self.word2weight = None
        #self.dim = len(list(word2vec.values())[0])
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)

        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda : max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    #or [np.zeros(300)], axis = 0)
                    or [np.zeros(self.dim)], axis = 0 )
            for words in X

        ])

def make_word2vecs_from_doc(text):
    #Tar en string, finner word2vec representasjon for hver av ordene, returnerer vector som inneholder
    # den opprinnelige stringen represenetert som word2vec for hvert ord. Altså [word2vec(ord1), word2vec(ord2)....]
    word2vecDoc = []
    for word in text.split():
        try:
             #print(model[word.lower()])
             word2vecDoc.append(w2v_model[word.lower()])
        except KeyError:
            pass
    return word2vecDoc

def get_articles(original_name):
    # Tar inn en textfil som er labelet på fasttext-format. Gir ut to arrays. Et med deweys og et med tekstene. [deweys],[texts]
    articles=open(original_name+'.txt',"r")
    articles=articles.readlines()
    dewey_array = []
    docs = []
    dewey_dict = {}
    for article in articles:
        dewey=article.partition(' ')[0].replace("__label__","")
        article_label_removed = article.replace("__label__"+dewey,"")
        docs.append(article_label_removed)
        dewey_array.append(dewey)

    return dewey_array, docs


def get_articles_from_folder(folder):
    ''' Tar inn en textfil som er labelet på fasttext-format. Gir ut 3 arrays. Et med navn på tekstfilene,
    et med deweys og et med tekstene. [tekst_navn][deweys],[texts]'''
    arr = os.listdir(folder)
    arr_txt = [path for path in os.listdir(folder) if path.endswith(".txt")]


    dewey_array = []
    docs = []
    for article_path in arr_txt:
            article = open(os.path.join(folder,article_path), "r")
            article = article.readlines()
            for article_content in article:
                dewey=article_content.partition(' ')[0].replace("__label__","")
                text  = article_content.replace("__label__"+dewey,"")
                docs.append(text)
                dewey_array.append(dewey[:3])
    text_names = [path.replace('.txt','') for path in arr_txt ]
    #print(len(dewey_array))
    #print(text_names)
    return text_names, dewey_array, docs

def logReg(x_train, y_train, vectorization_type):

    if vectorization_type =="tfidf":
       model = logReg_tfidf(x_train, y_train)
    elif vectorization_type =="countVectorization":
       model = logReg_Count(x_train, y_train)
    else:
        print("Vectorization type is not existing. Alternatives: tfidf or count")
        model = None
    return model

def logReg_tfidf(x_train, y_train):
    logres_tfidf_model = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("logres_tfidf", LogisticRegression())])
    logres_tfidf_model.fit(x_train, y_train)
    return logres_tfidf_model

def logReg_Count(x_train, y_train):
    logres_model = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("logres", LogisticRegression())])
    logres_model.fit(x_train, y_train)
    return logres_model

def svm(x_train, y_train, vectorization_type, w2v_path = None):
    model = None
    if vectorization_type == "tfidf":
        model = svm_tfidf(x_train, y_train)
    elif vectorization_type == "mean_embedding":
        if w2v_path is not None:
            word2vec_model = gensim.models.Doc2Vec.load(w2v_path)
            model = svm_meanembedding(x_train, y_train, word2vec_model)
        else:
            print("word2vec modell finnes ikke. Test avsluttes")
            model= None
    elif vectorization_type == "count":
        model = svm_count(x_train, y_train)
    else:
        print("Vectorization type is not existing. Alternatives: tfidf,count or meanembedding")
        model = None

    return model
def svm_tfidf(x_train, y_train):

    SVM_tfidf= Pipeline([('tfidf_vectorizer', TfidfVectorizer(analyzer= lambda x: x)), ('linear_svc', SVC(kernel ="linear"))])
    SVM_tfidf.fit(x_train,y_train)
    return SVM_tfidf

def svm_count(x_train, y_train):

    SVM_count= Pipeline([('count_vectorizer', CountVectorizer(analyzer= lambda x: x)), ('linear_svc', SVC(kernel ="linear"))])
    SVM_count.fit(x_train,y_train)
    return SVM_count

def svm_meanembedding(x_train,y_train, word2vec_model):

    SVC_model_pipe = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(word2vec_model)),
                          ("SVM", SVC())])
    SVC_model_pipe.fit(x_train,y_train)
    return SVC_model_pipe

def multiNomialBayes(x_train, y_train, vectorization_type):
    model = None
    if vectorization_type == "tfidf":
        model = multiNomialBayes_tfidf(x_train, y_train)
    elif vectorization_type == "count":
        model = multiNomialBayes_count(x_train, y_train)
    return model
def multiNomialBayes_tfidf(x_train, y_train):

    mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    mult_nb_tfidf.fit(x_train,y_train)
    return mult_nb_tfidf
def multiNomialBayes_count(x_train, y_train):
    mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    mult_nb.fit(x_train,y_train)

    return mult_nb

def getPredictionsAndAccuracy(x_test, y_test, model, returnAccuracy):
    predictions = []
    accuracy = 0
    if len(x_test) > 0 and len(y_test) >0 and model is not None:
        for text in x_test:
            predictions.append(model.predict([text]))
        if returnAccuracy == True:
            accuracy = accuracy_score(y_test,predictions)
    else:
        print("Input var ikke riktig. Sjekk om modell og  testsett eksisterer")
    return predictions, accuracy

def print_results(testName,res_vector,dewey_test):

    return "Results "+testName+": "+str(accuracy_score(dewey_test,res_vector))

def saveSklearnModels(model, output_filename):
    model_save_file = open(output_filename,'wb')
    dill.dump(model, model_save_file, -1)
    print("modell_lagret")
def loadSklearnModel(modelpicklePath):
    open_pickle_model_object = open(modelpicklePath, "rb")
    model = dill.load(open_pickle_model_object)
    return model
if __name__ == '__main__':

    w2v_model_path = "w2v_tgc/full.bin"
    dewey_train, text_train = get_articles("corpus_w_wiki/data_set_100/combined100_training")
    dewey_train = dewey_train[:10]
    text_train = text_train[:10]
    #dewey_test , text_test = get_articles("test_min500")

    #text_names, dewey_train, text_train = get_articles_from_folder("corpus_w_wiki/data_set_100/100_test")
    dewey_test, text_test = get_articles("corpus_w_wiki/data_set_100/100_test")

    model= multiNomialBayes(text_train, dewey_train,"count")

    predictions, accuracy = getPredictionsAndAccuracy(x_test = text_test, y_test = dewey_test, model = model, returnAccuracy = True)
    #print(predictions)
    print(accuracy)
    saveSklearnModels(model, "save_test.pckl")
    model_loaded = loadSklearnModel("save_test.pckl")
    predictions, accuracy = getPredictionsAndAccuracy(x_test=text_test, y_test=dewey_test, model=model_loaded,returnAccuracy=True)
    #print(predictions)
    print(accuracy)



    #dewey_train, text_train = get_articles("corpus_w_wiki/Datasett_100_w_wiki/train_w_wiki100")


    # ## TEST 1 Etrees med tfidf
    # etree_tfidf_model_pipe = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v_model)),
    #                       ("extra trees", ExtraTreesClassifier(n_estimators=400))])
    # print("Etree-modellen er produsert")
    # etree_tfidf_model_pipe.fit(text_train,dewey_train)
    # print("E-tree Modellen er trent. Predikering pågår.")
    # i = 0
    # etree_tfidf_results = []
    #
    #
    # # TEST 2 SVC + embeddings

    #
    #
    # ## TEST 3 SVM med TFIDF, uten embeddings
    #

    # #Test 4 Multinomial Naive Bayes
    # print("Test 4 Bernoulli Naive Bayes - Done")
    # # Test 5 Bernoulli nb med count vectorizer
    # bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    # bern_nb.fit(text_train, dewey_train)
    # bern_nb_res = []
    # print("Test 5  Bernoulli nb med count vectorizer - Done")
    # # Test 5 multinomial bayes med tfidf

    # print("Test 5 bernoulli bayes med tfidf bernoulli - Done")
    # # Test 6 bernoulli naive bayes med tfidf
    # bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    # bern_nb_tfidf.fit(text_train,dewey_train)
    # bern_nb_tfidf_res = []
    # print("Test 6 bernoulli naive bayes med tfidf - Done")


    # for article in text_test:
        #etree_results.append(etree_model_pipe.predict([article]))
        # etree_tfidf_results.append(etree_tfidf_model_pipe.predict([article]))


        # bern_nb_res.append(bern_nb.predict([article]))
        # bern_nb_tfidf_res.append(bern_nb_tfidf.predict([article]))



    # print(print_results("Etree-embedding", etree_results, dewey_test))
    # print(print_results("Etree-embedding w/tfidf",etree_tfidf_results,dewey_test))

    # print(print_results("Bernoulli Naive Bayes", bern_nb_res, dewey_test))
    # print(print_results("Bernoulli Naive Bayes w/tfidf", bern_nb_tfidf_res, dewey_test))


