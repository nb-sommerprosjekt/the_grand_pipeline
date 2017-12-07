from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras import backend as K
import numpy as np
import gensim
from keras.layers import Embedding
from keras.models import Model, load_model
import datetime
import time
import MLP
import os
import pickle
import re
import evaluator2
import utils
from sklearn.metrics import accuracy_score

def fasttextTrain2CNN(training_set, max_sequence_length, vocab_size):
    '''Transforming training set from fasttext format to CNN format.'''
    dewey_train, text_train = utils.get_articles(training_set)
    labels_index = {}
    labels = []
    for dewey in set(dewey_train):
        label_id = len(labels_index)
        labels_index[dewey] = label_id
    for dewey in dewey_train:
        labels.append(labels_index[dewey])

    num_classes = len(set(dewey_train))

    tokenizer = Tokenizer(num_words= vocab_size)
    tokenizer.fit_on_texts(text_train)
    sequences = tokenizer.texts_to_sequences(text_train)

    #print(sequences)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    x_train = pad_sequences(sequences, maxlen=max_sequence_length)

    y_train = to_categorical(np.asarray(labels))

    return x_train, y_train, word_index, labels_index, tokenizer, num_classes

def create_embedding_matrix(word2vecModel,word_index):
    '''Creating embedding matrix from words in vocabulary'''
    EMBEDDING_DIM = 100
    w2v_model = gensim.models.Doc2Vec.load(word2vecModel)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    j=0
    k=0
    for word, i in word_index.items():
        k = k+1
        try:
            if w2v_model[word] is not None:
            # words not found in embedding index will be all-zeros.

                embedding_matrix[i] = w2v_model[word]
        except KeyError:
            j=j+1
            continue
    return embedding_matrix, EMBEDDING_DIM

def train_cnn(training_set, VOCAB_SIZE, MAX_SEQUENCE_LENGTH,EPOCHS, FOLDER_TO_SAVE_MODEL, loss_model,
              VALIDATION_SPLIT, word2vec_file_name):
    '''Training embedded cnn model'''
    start_time = time.time()

    #word2vec = word2vec_file_name

    x_train, y_train, word_index, labels_index, tokenizer, num_classes = fasttextTrain2CNN(training_set=training_set, max_sequence_length=MAX_SEQUENCE_LENGTH,vocab_size= VOCAB_SIZE)
    embedding_matrix, EMBEDDING_DIM  = create_embedding_matrix(word2vec_file_name,word_index)

    sequence_input = Input(shape = (MAX_SEQUENCE_LENGTH,), dtype = 'int32')
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)

    x = Conv1D(128, 5, activation = 'relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation = 'relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128,5, activation = 'relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)

    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss=loss_model,
                  optimizer="rmsprop",
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=EPOCHS
              )

    time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

    if not os.path.exists(FOLDER_TO_SAVE_MODEL):
        os.makedirs(FOLDER_TO_SAVE_MODEL)
    model_directory = os.path.join(FOLDER_TO_SAVE_MODEL , "cnn-" + str(VOCAB_SIZE) + "-" + str(MAX_SEQUENCE_LENGTH) + "-" + str(
        EPOCHS) + "-" + str(time_stamp))
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    #folder_to_save_model= MODEL_DIRECTORY+str(VOCAB_SIZE)+"-"+str(MAX_SEQUENCE_LENGTH)+"-"+str(EPOCHS)+"-"+str(time_stamp)

    TIME_ELAPSED = time.time() - start_time


    save_model_path = model_directory+"/model.bin"
    utils.log_model_stats(model_directory = model_directory , training_set_name = training_set
                         ,training_set = x_train,num_classes = num_classes, vocab_size = VOCAB_SIZE,
                         max_sequence_length= MAX_SEQUENCE_LENGTH, epochs=EPOCHS, time_elapsed = TIME_ELAPSED,
                         path_to_model= save_model_path, loss_model =loss_model,  vectorization_type= None,
                         validation_split = VALIDATION_SPLIT, word2vec = word2vec_file_name)

    #Saving model
    model.save(save_model_path)
    K.clear_session()
    print("modell er nå lagret i folder: {}".format(model_directory))

    #Saving tokenizer
    with open(model_directory+'/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #Saving indexes
    with open(model_directory+'/label_indexes.pickle', 'wb') as handle:
        pickle.dump(labels_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model_directory

def cnn_pred(test_set, mod_dir, k_top_labels, isMajority_rule=True):
    '''Test module for CNN'''
    dewey_test, text_test = utils.get_articles(test_set)
    #Loading model
    model = load_model(mod_dir+'/model.bin')

    # loading tokenizer
    with open(mod_dir+'/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # loading label indexes
    with open(mod_dir + '/label_indexes.pickle', 'rb') as handle:
        labels_index = pickle.load(handle)

    # Loading parameters like max_sequence_length, vocabulary_size and vectorization_type
    with open(mod_dir+'/model_stats', 'r') as params_file:
        params_data = params_file.read()


    print('Found {} unique tokens.'.format(len(test_word_index)))


    re_max_seq_length = re.search('length:(.+?)\n', params_data)
    if re_max_seq_length:
            MAX_SEQUENCE_LENGTH = int(re_max_seq_length.group(1))
            print("Max sequence length: {}".format(MAX_SEQUENCE_LENGTH))
    re_vocab_size = re.search('size:(.+?)\n', params_data)
    if re_vocab_size:
        vocab_size = int(re_vocab_size.group(1))
        print("The vocabulary size: {}".format(vocab_size))
    if isMajority_rule == True:
        predictions, test_accuracy = cnn_majority_rule_test(test_set_dewey=test_set, MODEL=model,
                                                            MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH,
                                                            TRAIN_TOKENIZER = tokenizer, LABEL_INDEX_VECTOR = labels_index,
                                                            k_output_labels=k_top_labels)

    else:
        test_labels = []
        for dewey in dewey_test:
             test_labels.append(labels_index[dewey])
        test_sequences = tokenizer.texts_to_sequences(text_test)
        test_word_index = tokenizer.word_index
        x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

        y_test = to_categorical(test_labels)

        test_score, test_accuracy = model.evaluate(x_test, y_test, batch_size= 64, verbose=1)
        print('Test_score:', str(test_score))
        print('Test Accuracy', str(test_accuracy))
        #k_top_labels=3
        predictions = utils.prediction(model, x_test, k_top_labels,labels_index)

    #Writing results to txt-file.
    with open(mod_dir+"/result.txt",'a') as result_file:
        result_file.write('test_set:'+test_set+'\n'+
                          #'Test_score:'+ str(test_score)+ '\n'
                          'Test_accuracy:' + str(test_accuracy)+'\n\n')
    return predictions

def cnn_majority_rule_test(test_set_dewey,MODEL,MAX_SEQUENCE_LENGTH, TRAIN_TOKENIZER, LABEL_INDEX_VECTOR
                           ,k_output_labels):
    total_preds =[]
    y_test_total = []
    one_pred = []
    for i in range(len(test_set_dewey)):


            dewey =test_set_dewey[i][0]

            texts = test_set_dewey[i][1]
            dewey_label_index= LABEL_INDEX_VECTOR[dewey.strip()]
            y_test = []
            new_texts =[]

            for j in range(0, len(texts)):
                y_test.append(dewey_label_index)
                new_texts.append(' '.join(texts[j]))

            test_sequences = TRAIN_TOKENIZER.texts_to_sequences(new_texts)
            #test_sequences_matrix = TRAIN_TOKENIZER.sequences_to_matrix(test_sequences, mode=VECTORIZATION_TYPE)
            x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
            y_test = to_categorical(np.asarray(y_test))


            predictions = utils.prediction(MODEL, x_test, k_output_labels, LABEL_INDEX_VECTOR)
            y_test_total.append(dewey)
            majority_rule_preds = evaluator2.majority_rule(predictions,k_output_labels)
            total_preds.append(majority_rule_preds)
            one_pred.append(majority_rule_preds[0])

    accuracy = accuracy_score(y_test_total, one_pred)
    return total_preds, accuracy

def run_cnn_tests(TRAINING_SET, TEST_SET, VOCAB_VECTOR, SEQUENCE_LENGTH_VECTOR, EPOCHS, FOLDER_TO_SAVE_MODEL, LOSS_MODEL,
                  validation_split, word2vec_model, k_top_labels):
    '''Module for running test sequences with different parameters.'''

    validation_split=float(validation_split)


    if isinstance(VOCAB_VECTOR,str):
        VOCAB_VECTOR=[int(VOCAB_VECTOR)]
    else:
        VOCAB_VECTOR= list(map(int, VOCAB_VECTOR))

    if isinstance(SEQUENCE_LENGTH_VECTOR,str):
        SEQUENCE_LENGTH_VECTOR=[int(SEQUENCE_LENGTH_VECTOR)]
    else:
        SEQUENCE_LENGTH_VECTOR= list(map(int, SEQUENCE_LENGTH_VECTOR))

    if isinstance(EPOCHS,str):
        EPOCHS=[int(EPOCHS)]
    else:
        EPOCHS= list(map(int, EPOCHS))

        k_top_labels=int(k_top_labels)


    for vocab_test in VOCAB_VECTOR:
        for sequence_length_test in SEQUENCE_LENGTH_VECTOR:
            for epoch_test in EPOCHS:
                run_training=True
                if run_training:
                    test_mod_dir=train_cnn(TRAINING_SET,
                                               VOCAB_SIZE=vocab_test,
                                               MAX_SEQUENCE_LENGTH=sequence_length_test,
                                               EPOCHS=epoch_test,
                                               FOLDER_TO_SAVE_MODEL=FOLDER_TO_SAVE_MODEL,
                                               loss_model=LOSS_MODEL,
                                               VALIDATION_SPLIT= validation_split,
                                               word2vec_file_name= word2vec_model
                                               )

                try:
                    cnn_pred(TEST_SET, test_mod_dir, k_top_labels)
                except ValueError:

                    print("Noe gikk galt, prøver gjenkjenning på nytt.")
                    cnn_pred(TEST_SET, test_mod_dir, k_top_labels)

# if __name__ == '__main__':
#     vocab_vector = [5000]
#     sequence_length_vector = [5000]
#     epoch_vector = [1]
#     run_cnn_tests(TRAINING_SET= "corpus_w_wiki/data_set_100/combined100_training", TEST_SET= "corpus_w_wiki/data_set_100/100_test", VOCAB_VECTOR=vocab_vector
#                    , SEQUENCE_LENGTH_VECTOR= sequence_length_vector,EPOCHS= epoch_vector, FOLDER_TO_SAVE_MODEL = "cnn/",
#                    LOSS_MODEL= "categorical_crossentropy", validation_split= 'None', word2vec_model ="w2v_tgc/full.bin", k_top_labels = 5)

    #cnn_pred("corpus_w_wiki/data_set_100/100_test", 'cnn/cnn-5000-5000-10-20171110130602', 5)