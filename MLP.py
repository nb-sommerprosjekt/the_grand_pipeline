# INSPIRERT av MLP fra http://nadbordrozd.github.io/blog/2017/08/12/looking-for-the-text-top-model/

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation
import numpy as np
from keras.models import Model, Sequential, load_model
import time
import os
import datetime
import pickle

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


def train_mlp(TRAINING_SET, TEST_SET, BATCH_SIZE, VOCAB_SIZE, MAX_SEQUENCE_LENGTH,EPOCHS, FOLDER_TO_SAVE_MODEL, LOSS_MODEL, MODEL_STATS_FILE_NAME):
    '''Creating model'''

    start_time = time.time()

    ## Preprocessing
    vocab_size =VOCAB_SIZE


    x_train, y_train, tokenizer, num_classes, labels_index = fasttextTrain2mlp(TRAINING_SET, TEST_SET, MAX_SEQUENCE_LENGTH, vocab_size)
    x_test, y_test = fasttextTest2mlp(TEST_SET,MAX_SEQUENCE_LENGTH,vocab_size, tokenizer, labels_index)
    ####Preparing test_set

    model = Sequential()
    model.add(Dense(128, input_shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, input_shape=(vocab_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    model.summary()
    model.compile(loss=LOSS_MODEL,
                  optimizer='adam',
                  metrics=['accuracy'])


    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1
              )

    #Lagre modell
    model_time_stamp = '{:%Y%m%d%H%M}'.format(datetime.datetime.now())
    model_directory = FOLDER_TO_SAVE_MODEL + "mlp-" + str(vocab_size) + "-" + str(MAX_SEQUENCE_LENGTH) + "-" + str(
        EPOCHS) + "-" + str(model_time_stamp)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)


    save_model_path = model_directory+"/model.bin"
    model.save(save_model_path)

    #Lagre test-fil
    time_elapsed = time.time() - start_time
    log_model_stats(MODEL_STATS_FILE_NAME, model_directory,TRAINING_SET,x_train
                ,num_classes, vocab_size, MAX_SEQUENCE_LENGTH
                , EPOCHS, time_elapsed, save_model_path, LOSS_MODEL)

    #Lagre tokenizer
    with open(model_directory+'/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #Lagre label_indexes
    with open(model_directory+'/label_indexes.pickle', 'wb') as handle:
        pickle.dump(labels_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #Skrive nøkkelparametere til tekstfil
    with open(model_directory+'/stats.txt','w') as statsfile:
        statsfile.write(str(MAX_SEQUENCE_LENGTH)+","+str(vocab_size))

    print("Modell ferdig trent og lagret i "+ model_directory)
    return model_directory
def test_mlp(TEST_SET, MODEL_DIRECTORY, MAX_SEQUENCE_LENGTH,vocab_size):

    #Loading model
    model = load_model(MODEL_DIRECTORY+'/model.bin')

    # loading tokenizer
    with open(MODEL_DIRECTORY+'/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # loading label indexes
    with open(MODEL_DIRECTORY + '/label_indexes.pickle', 'rb') as handle:
        labels_index = pickle.load(handle)


    x_test, y_test = fasttextTest2mlp(TEST_SET, MAX_SEQUENCE_LENGTH, vocab_size, tokenizer, labels_index)


    test_score,test_accuracy = evaluation(model,x_test,y_test, VERBOSE = 1)
    print('Test_score:', test_score)
    print('Test Accuracy', test_accuracy)
    with open(MODEL_DIRECTORY+"/result.txt",'a') as result_file:
        result_file.write('Test_score:'+ str(test_score)+ '\n'
                          'Test_accuracy:' + str(test_accuracy)+'\n')

    return test_score,test_accuracy
def fasttextTrain2mlp(FASTTEXT_TRAIN_FILE,FASTTEXT_TEST_FILE,MAX_SEQUENCE_LENGTH, VOCAB_SIZE):
    '''Converting training_set from fasttext format to MLP-format'''

    dewey_train, text_train = get_articles(FASTTEXT_TRAIN_FILE)
    dewey_test, text_test = get_articles(FASTTEXT_TEST_FILE)

    ## Preprocessing training_set
    labels_index = {}
    labels = []
    for dewey in set(dewey_train):
        label_id = len(labels_index)
        labels_index[dewey] = label_id
    for dewey in dewey_train:
        labels.append(labels_index[dewey])
    print(len(labels_index))
    print(labels_index)
    print(len(labels))
    num_classes = len(set(dewey_train))
    #Preparing_training_set
    tokenizer = Tokenizer(num_words= VOCAB_SIZE)
    tokenizer.fit_on_texts(text_train)
    sequences = tokenizer.texts_to_sequences(text_train)
    sequence_matrix = tokenizer.sequences_to_matrix(sequences, mode = 'binary')

    data = pad_sequences(sequence_matrix, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))

    print(labels.shape)


    x_train = data
    y_train = labels


    return x_train, y_train, tokenizer, num_classes,labels_index #x_test, y_test, num_classes

def fasttextTest2mlp(FASTTEXT_TEST_FILE,MAX_SEQUENCE_LENGTH, VOCAB_SIZE, TRAIN_TOKENIZER, LABEL_INDEX_VECTOR):
    # Preparing test data
    dewey_test, text_test = get_articles(FASTTEXT_TEST_FILE)
    test_labels = []

    for dewey in dewey_test:
       test_labels.append(LABEL_INDEX_VECTOR[dewey.strip()])

    test_sequences = TRAIN_TOKENIZER.texts_to_sequences(text_test)
    test_sequence_matrix = TRAIN_TOKENIZER.sequences_to_matrix(test_sequences, mode = 'binary')
    test_word_index = TRAIN_TOKENIZER.word_index
    print('Found %s unique tokens.' % len(test_word_index))
    x_test = pad_sequences(test_sequence_matrix, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = to_categorical(np.asarray(test_labels))

    return  x_test, y_test

def evaluation(MODEL, X_TEST,Y_TEST, VERBOSE):
        '''Evaluates model. Return accuracy and score'''
        score = MODEL.evaluate(X_TEST, Y_TEST, VERBOSE)
        test_score = score[0]
        test_accuracy = score[1]
        return  test_score, test_accuracy

def log_model_stats(model_stats_file_name, model_directory, training_set_name, training_set,
                 num_classes,vocab_size, max_sequence_length,
                epochs, time_elapsed,
                path_to_model, loss_model ):
    ''' Prints results and parameters to log-file.'''

    time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_stats_file = open(model_directory+"/"+model_stats_file_name, "a")
    model_stats_file.write("Time:" + str(time_stamp) + '\n'
                      + "Time elapsed: " + str(time_elapsed) + '\n'
                      + "Training set:" + training_set_name + "\n"
                      + "Vocab_size:" + str(vocab_size) + '\n'
                      + "Max_sequence_length:" + str(max_sequence_length) + '\n'
                      + "Epochs:" + str(epochs) + '\n'
                      + "Antall deweys:" + str(num_classes) + '\n'
                      + "Antall docs i treningssett:" + str(len(training_set)) + '\n'
                      + "saved_model:" + path_to_model + '\n'
                      + "loss_model:" + str(loss_model) + '\n' + '\n'
                      )
    model_stats_file.close()





def run_mlp_tests (training_set, test_set, save_model_folder,model_stats_file_name,
                   batch_size,vocab_size_vector, sequence_length_vector, epoch_vector, loss_model):

    for vocab_test in vocab_size_vector:
        for sequence_length_test in sequence_length_vector:
            for epoch_test in epoch_vector:



                MOD_DIR=train_mlp(TRAINING_SET = training_set,
                        TEST_SET =test_set,
                        BATCH_SIZE=batch_size,
                        VOCAB_SIZE=vocab_test,
                        MAX_SEQUENCE_LENGTH =sequence_length_test,
                        EPOCHS=epoch_test,
                        FOLDER_TO_SAVE_MODEL=save_model_folder,
                        LOSS_MODEL= loss_model,
                        MODEL_STATS_FILE_NAME = model_stats_file_name,
                        )

                print("Prosessen fortsetter")
                try:
                    test_mlp(test_set,MOD_DIR,sequence_length_test,vocab_test)
                except TypeError:
                     print("Noe gikk feil med testen")

if __name__ == '__main__':

    run_mlp_tests("corpus_w_wiki/data_set_100/combined100_training", "corpus_w_wiki/data_set_100/100_test",save_model_folder= "mlp/",
                   model_stats_file_name= "model_stats",batch_size=64, vocab_size_vector=[5000], sequence_length_vector = [5000]
                   ,epoch_vector = [10], loss_model = "categorical_crossentropy")
    #test_mlp("corpus_w_wiki/data_set_100/100_test","mlp/mlp-5000-5000-10-201711081658",5000,5000)