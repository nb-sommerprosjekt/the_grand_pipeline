import os
import data_prep
import sys
from MLP import run_mlp_tests
from CNN import run_cnn_tests
from data_augmentation import create_fake_corpus
import nltk


if __name__ == '__main__':
    nltk.download('omw')

    print("Trying to load config-file named: {}".format(sys.argv[1]))
    config=data_prep.load_config(sys.argv[1])

    corpus_folder=data_prep.preprocess(config["name_corpus"],config["data_set"],config["data_set_folder"], config["stemming"], config["stop_words"],config["sentences"], config["lower_case"], config["extra_functions"])
    #corpus_folder = "data_set/test_torsdag_folder"
    wiki_corpus_folder=data_prep.preprocess_wiki(corpus_folder,config["name_corpus"]+"_wiki", config["stemming"], config["stop_words"], config["sentences"],config["lower_case"], config["extra_functions"])
    training_folder, test_folder=data_prep.split_to_training_and_test(corpus_folder, config["name_corpus"],0.2,3)
    if config["wikipedia"]:
        wiki_corpus_folder = data_prep.preprocess_wiki(corpus_folder, config["name_corpus"] + "_wiki", config["stemming"],
                                             config["stop_words"], config["sentences"], config["lower_case"],
                                             config["extra_functions"])

        data_prep.add_wiki_to_training(wiki_corpus_folder,training_folder)
    #load_config_file
    training_folder = "data_set/test_torsdag_folder/test_torsdag_test"
    test_folder = "data_set/test_torsdag_w_fakes_folder/test_torsdag_w_fakes_test"
    training_folder=data_prep.split_articles(training_folder,1000)
    if config["da_run"] == "True":
        artificial_training_folder = training_folder + "artificial"
        if os.path.exists(artificial_training_folder):
            print("Artificial set already works. No action will be taken.")
        else:
            data_prep.create_folder(artificial_training_folder)
            create_fake_corpus(training_folder, artificial_training_folder, config["da_splits"], config["da_noise_percentage"],config["da_noise_method"])
        training_folder=artificial_training_folder

    rubbish_folder,valid_deweys,training_set_length=data_prep.remove_unecessary_articles(training_folder,corpus_folder,config["minimum_articles"],config["dewey_digits"])
    #print(valid_deweys)
    test_folder,test_set_length,dewey_and_texts=data_prep.prep_test_set(test_folder,valid_deweys,config["article_size"],config["dewey_digits"])

    test_text=data_prep.load_set(test_folder)
    training_text=data_prep.load_set(training_folder)
    test_file=data_prep.save_file("tmp","test_file.txt",test_text)
    training_file=data_prep.save_file("tmp","training_file.txt",training_text)
    data_prep.create_folder(os.path.join("fasttext",config["ft_run_name"]))

    if not (os.path.exists(os.path.join(corpus_folder, "run_log.txt"))):
        parameters= open(os.path.join("config",sys.argv[1]+".txt"),"r")
        parameters=parameters.read()
        logfile=open(os.path.join(corpus_folder, "run_log.txt"), "w")
        logfile.write("Copy of the parameters used when this set was created: \n\n")
        logfile.write(parameters+"\n")
        logfile.write("The number of articles in the training set: {}\n".format(training_set_length))
        logfile.write("The number of articles in the test set: {}\n".format(test_set_length))
        dewey_list=list(valid_deweys)
        dewey_list.sort()
        logfile.write("Here is the deweys in the training: \n")
        logfile.write(str(dewey_list))


    #if config["ft_run"]=="True":
    #    train_fasttext_models(training_text,test_text,os.path.join("fasttext",config["ft_run_name"]),config["ft_epochs"],config["ft_lr"],config["ft_lr_update"],config["ft_word_window"],config["ft_loss"],config["ft_wiki_vec"],config["ft_k_labels"],config["minimum_articles"],config["dewey_digits"],config["ft_save_model"],config["ft_top_k_labels"],dewey_and_texts)
    if config["mlp_run"]=="True":
        run_mlp_tests(training_file,dewey_and_texts,config["mlp_save_model_folder"],config["mlp_batch_size"],config["mlp_vocab_size_vector"],config["mlp_sequence_length_vector"],config["mlp_epoch_vector"],config["mlp_loss_model"],config["mlp_vectorization_type"],config["mlp_validation_split"],config["mlp_k_labels"])
    if config["cnn_run"] == "True":
        run_cnn_tests(training_file,dewey_and_texts,config["cnn_vocab_size_vector"],config["cnn_sequence_length_vector"],config["cnn_epoch_vector"],config["cnn_save_model_folder"],config["cnn_loss_model"],config["cnn_validation_split"],config["cnn_w2v"],config["cnn_k_labels"])
