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
import re
import evaluator2
import utils
from sklearn.metrics import accuracy_score


def train_mlp(TRAINING_SET, BATCH_SIZE, VOCAB_SIZE, MAX_SEQUENCE_LENGTH,EPOCHS, FOLDER_TO_SAVE_MODEL, LOSS_MODEL, VECTORIZATION_TYPE, VALIDATION_SPLIT):
    '''Training model'''

    start_time = time.time()

    ## Preprocessing
    vocab_size =VOCAB_SIZE
    VALIDATION_SPLIT=float(VALIDATION_SPLIT)
    EPOCHS=int(EPOCHS)
    BATCH_SIZE=int(BATCH_SIZE)
    VOCAB_SIZE=int(VOCAB_SIZE)

    x_train, y_train, tokenizer, num_classes, labels_index = fasttextTrain2mlp(TRAINING_SET, MAX_SEQUENCE_LENGTH, vocab_size
                                                                               ,VECTORIZATION_TYPE, folder = False)

    ####Preparing test_set

    model = Sequential()
    model.add(Dense(512, input_shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, input_shape=(vocab_size,)))
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
              verbose=1,
              validation_split= VALIDATION_SPLIT
              )

    #Lagre modell
    model_time_stamp = '{:%Y%m%d%H%M}'.format(datetime.datetime.now())
    model_directory = os.path.join(FOLDER_TO_SAVE_MODEL ,"mlp-" + str(vocab_size) + "-" + str(MAX_SEQUENCE_LENGTH) + "-" + str(
        EPOCHS) + "-" + str(model_time_stamp))
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)


    save_model_path = os.path.join(model_directory,"model.bin")
    model.save(save_model_path)


    time_elapsed = time.time() - start_time
    # Skrive nøkkelparametere til tekstfil
    utils.log_model_stats(model_directory,TRAINING_SET,x_train
                ,num_classes, vocab_size, MAX_SEQUENCE_LENGTH
                , EPOCHS, time_elapsed, save_model_path,
                    LOSS_MODEL, VECTORIZATION_TYPE,VALIDATION_SPLIT, word2vec= None)

    #Lagre tokenizer
    with open(model_directory+'/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #Lagre label_indexes
    with open(model_directory+'/label_indexes.pickle', 'wb') as handle:
        pickle.dump(labels_index, handle, protocol=pickle.HIGHEST_PROTOCOL)



    print("Modell ferdig trent og lagret i "+ model_directory)
    return model_directory

def test_mlp(TEST_SET, MODEL_DIRECTORY, k_output_labels, isMajority_rule=True):
    #TEST_SET = deweys_and_texts
    '''Test module for MLP'''
    #format of test_set [[dewey][text_split1, text_split2,text_split3]]
    #Loading model
    model = load_model(os.path.join(MODEL_DIRECTORY,'model.bin'))

    # loading tokenizer
    with open(os.path.join(MODEL_DIRECTORY,"tokenizer.pickle"), 'rb') as handle:
        tokenizer = pickle.load(handle)
    # loading label indexes
    with open(os.path.join(MODEL_DIRECTORY ,"label_indexes.pickle"), 'rb') as handle:
        labels_index = pickle.load(handle)

    # Loading parameters like max_sequence_length, vocabulary_size and vectorization_type
    with open(os.path.join(MODEL_DIRECTORY,"model_stats"), 'r') as params_file:
        params_data = params_file.read()

    re_max_seq_length = re.search('length:(.+?)\n', params_data)
    if re_max_seq_length:
            MAX_SEQUENCE_LENGTH = int(re_max_seq_length.group(1))
            print("Max sequence length:{}".format(MAX_SEQUENCE_LENGTH))
    re_vocab_size = re.search('size:(.+?)\n', params_data)
    if re_vocab_size:
        vocab_size = int(re_vocab_size.group(1))
        print("Vocabulary size: {}".format(vocab_size))

    re_vectorization_type = re.search('type:(.+?)\n',params_data)
    if re_vectorization_type:
        vectorization_type = re_vectorization_type.group(1)
        print("This utilizes the vectorization: {}".format(str(vectorization_type)))



    if isMajority_rule == True:
        predictions, test_accuracy= mlp_majority_rule_test(test_set_dewey=TEST_SET,MODEL=model,
                                                                       MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                                                       TRAIN_TOKENIZER = tokenizer,
                                                                       LABEL_INDEX_VECTOR = labels_index,
                                                                       VECTORIZATION_TYPE=vectorization_type,k_output_labels= k_output_labels)

    else:
        x_test, y_test = fasttextTest2mlp(TEST_SET, MAX_SEQUENCE_LENGTH, vocab_size, tokenizer, labels_index, VECTORIZATION_TYPE= vectorization_type)
        #test_score,test_accuracy = evaluation(model,x_test,y_test, VERBOSE = 1)
        predictions = utils.prediction(model, x_test, k_output_labels, labels_index)

        #print('Test_score:', test_score)
        #print('Test Accuracy', test_accuracy)
    # Writing results to txt-file.
    with open(os.path.join(MODEL_DIRECTORY,"result.txt"),'a') as result_file:
        result_file.write('Test_accuracy:' + str(test_accuracy)+'\n\n')
    return predictions


def fasttextTrain2mlp(FASTTEXT_TRAIN_FILE,MAX_SEQUENCE_LENGTH, VOCAB_SIZE, VECTORIZATION_TYPE, folder):
    '''Converting training_set from fasttext format to MLP-format'''
    if folder ==False:
        dewey_train, text_train = utils.get_articles(FASTTEXT_TRAIN_FILE)
    else:
        text_names, dewey_train, text_train = utils.get_articles_from_folder(FASTTEXT_TRAIN_FILE)

    labels_index = {}
    labels = []
    for dewey in set(dewey_train):
        label_id = len(labels_index)
        labels_index[dewey] = label_id
    for dewey in dewey_train:
        labels.append(labels_index[dewey])
    print("length of labels indexes: {} ".format(len(labels_index)))
    #print(labels_index)
    print("Length of labels:{}".format(len(labels)))
    num_classes = len(set(dewey_train))
    #Preparing_training_set
    tokenizer = Tokenizer(num_words= VOCAB_SIZE)
    tokenizer.fit_on_texts(text_train)
    sequences = tokenizer.texts_to_sequences(text_train)
    sequence_matrix = tokenizer.sequences_to_matrix(sequences, mode = VECTORIZATION_TYPE)

    data = pad_sequences(sequence_matrix, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))

    print(labels.shape)


    x_train = data
    y_train = labels


    return x_train, y_train, tokenizer, num_classes,labels_index #x_test, y_test, num_classes

def fasttextTest2mlp(FASTTEXT_TEST_FILE,MAX_SEQUENCE_LENGTH, TRAIN_TOKENIZER, LABEL_INDEX_VECTOR, VECTORIZATION_TYPE):
    ''' Preparing test data for MLP training'''
    dewey_test, text_test = utils.get_articles(FASTTEXT_TEST_FILE)
    test_labels = []

    for dewey in dewey_test:
       test_labels.append(LABEL_INDEX_VECTOR[dewey.strip()])

    test_sequences = TRAIN_TOKENIZER.texts_to_sequences(text_test)
    test_sequence_matrix = TRAIN_TOKENIZER.sequences_to_matrix(test_sequences, mode = VECTORIZATION_TYPE)

    x_test = pad_sequences(test_sequence_matrix, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = to_categorical(np.asarray(test_labels))

    return  x_test, y_test
def mlp_majority_rule_test(test_set_dewey,MODEL,MAX_SEQUENCE_LENGTH, VOCAB_SIZE, TRAIN_TOKENIZER, LABEL_INDEX_VECTOR
                           , VECTORIZATION_TYPE, k_output_labels):
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
            test_sequences_matrix = TRAIN_TOKENIZER.sequences_to_matrix(test_sequences, mode=VECTORIZATION_TYPE)
            x_test = pad_sequences(test_sequences_matrix, maxlen=MAX_SEQUENCE_LENGTH)
            y_test = to_categorical(np.asarray(y_test))


            predictions = utils.prediction(MODEL, x_test, k_output_labels, LABEL_INDEX_VECTOR)
            y_test_total.append(dewey)
            majority_rule_preds = evaluator2.majority_rule(predictions,k_output_labels)
            total_preds.append(majority_rule_preds)
            one_pred.append(majority_rule_preds[0])

    accuracy = accuracy_score(y_test_total, one_pred)
    return total_preds, accuracy


def evaluation(MODEL, X_TEST,Y_TEST, VERBOSE):
        '''Evaluates model. Return accuracy and score'''
        score = MODEL.evaluate(X_TEST, Y_TEST, VERBOSE)
        test_score = score[0]
        test_accuracy = score[1]

        return  test_score, test_accuracy





def run_mlp_tests (training_set, test_set, save_model_folder,
                   batch_size,vocab_size_vector, sequence_length_vector, epoch_vector, loss_model, vectorization_type, validation_split, k_output_labels, isMajority_rule=True):

    if isinstance(vocab_size_vector,str):
        vocab_size_vector=[int(vocab_size_vector)]
    else:
        vocab_size_vector= list(map(int, vocab_size_vector))

    if isinstance(sequence_length_vector,str):
        sequence_length_vector=[int(sequence_length_vector)]
    else:
        sequence_length_vector= list(map(int, sequence_length_vector))

    if isinstance(vocab_size_vector,str):
        epoch_vector=[int(epoch_vector)]
    else:
        epoch_vector= list(map(int, epoch_vector))
    k_output_labels=int(k_output_labels)


    '''Function for running test and training with different combinations of vocab_size, sequence_lenghts and epochs'''
    for vocab_test in vocab_size_vector:
        for sequence_length_test in sequence_length_vector:
            for epoch_test in epoch_vector:



                MOD_DIR=train_mlp(TRAINING_SET = training_set,
                        BATCH_SIZE=batch_size,
                        VOCAB_SIZE=vocab_test,
                        MAX_SEQUENCE_LENGTH =sequence_length_test,
                        EPOCHS=epoch_test,
                        FOLDER_TO_SAVE_MODEL=save_model_folder,
                        LOSS_MODEL= loss_model,
                        VECTORIZATION_TYPE= vectorization_type,
                        VALIDATION_SPLIT = validation_split
                        )

                print("Setter igang test")
                try:
                     test_mlp(test_set,MOD_DIR, k_output_labels, isMajority_rule)
                except ValueError:

                      print("Noe gikk feil med testen, prøver på nytt")
                      test_mlp(test_set, MOD_DIR,  k_output_labels, isMajority_rule)


