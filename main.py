import os
import preprocess_text
import shutil
from shutil import copyfile
import math
from math import floor
from random import shuffle
import time
from nltk.corpus import stopwords
from nltk.stem import snowball
from nltk.tokenize import word_tokenize
from numpy import array_split

def load_config(config_file):
    print("Loading config file: {}".format(config_file))
    config={}

    with open("config/"+config_file+".txt","r") as file:
        for line in file.readlines():
            if len(line)>4:
                if line[0]!="#":
                    temp=line.split(":::")
                    config[temp[0]]=temp[1].strip()
    print("Config-file loaded")
    return config

def preprocess(corpus_name,stemming, stop_words,sentences,lower_case,extra_functions):
    counter=0
    rootdir="tgc"
    if not os.path.exists(corpus_name):
        os.makedirs(corpus_name)
        print("Preprocessing the corpus from the folder 'tcg'.")
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if str(file)[:5] == "meta-":
                    counter += 1
                    if counter % 100 == 0:
                        print("File {} out of {}".format(counter, len(files)/2))
                    f = open(os.path.join(rootdir, file), "r+")
                    for line in f.readlines():
                        if "dewey:::" in line:
                            dewey = line.split(":::")[1]
                            dewey=dewey.strip()

                    f =open(os.path.join(rootdir, file[5:]),"r")
                    text=f.read()
                    if sentences:
                        text=preprocess_text.remove_punctuation(text)
                    if stop_words:
                        preprocess_text.remove_stopwords(text)
                    if stemming:
                        norStem = snowball.NorwegianStemmer()
                        preprocess_text.stem_text(text, norStem)
                    if lower_case:
                        text=text.lower()
                    if extra_functions:
                        # add function calls here
                        # FUNCTION_CALL()
                        # FUNCTION_CALL()
                        # FUNCTION_CALL()
                        pass
                    file=open(os.path.join(corpus_name, file[5:]),"w")
                    file.write("__label__"+dewey+" "+text)
        print("Preprocessed corpus saved in the folder {}".format(corpus_name))
        return corpus_name  #Kanskje kjøre neste steg.



    else:
        print("Corpus is already created, using {} ".format(corpus_name))
        return corpus_name  #Kanskje kjøre neste steg.

def preprocess_wiki(wiki_corpus_name,stemming, stop_words, sentences,lower_case, extra_functions):

    rootdir="tgc_wiki"
    counter=0
    if not os.path.exists(wiki_corpus_name):
        os.makedirs(wiki_corpus_name)
        print("Preprocessing wiki data")
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                counter += 1
                if counter%100==0:
                    print("File {} out of {}".format(counter,len(files)))
                # print(subdir)
                f = open(os.path.join(rootdir, file), "r+")
                text = f.read()

                text=text.split(" ")
                dewey = text[0].replace("__label__", "")
                text=" ".join(text[1:])

                if sentences:
                    text = preprocess_text.remove_punctuation(text)

                if stop_words:
                    preprocess_text.remove_stopwords(text)

                if stemming:
                    norStem = snowball.NorwegianStemmer()
                    preprocess_text.stem_text(text, norStem)

                if lower_case:
                    text = text.lower()

                if extra_functions:
                    # add function calls here
                    # FUNCTION_CALL()
                    # FUNCTION_CALL()
                    # FUNCTION_CALL()
                    pass

                file = open(os.path.join(wiki_corpus_name, file), "w+")
                file.write("__label__" + dewey + " " + text)
        print("Preprocessed  wiki_corpus saved in the folder {}".format(wiki_corpus_name))
        return wiki_corpus_name  #Kanskje kjøre neste steg.

    else:
        print("Wiki corpus is already created, using {} ".format(wiki_corpus_name))
        return wiki_corpus_name  #Kanskje kjøre neste steg.



def split_to_training_and_test(corpus_name, test_ratio,dewey_digits):
    print("Splitting to training and test:")
    training_folder="data_set/"+corpus_name+"_training"
    test_folder="data_set/"+corpus_name+"_test"

    if  os.path.exists(training_folder):
        shutil.rmtree(training_folder)
        os.makedirs(training_folder)
    else:
        os.makedirs(training_folder)

    if  os.path.exists(test_folder):
        shutil.rmtree(test_folder)
        os.makedirs(test_folder)
    else:
        os.makedirs(test_folder)

    dewey_dict={}


    for subdir, dirs, files in os.walk(corpus_name):
        for file in files:
            f = open(os.path.join(corpus_name, file), "r+")
            text = f.read()

            dewey = text.split(" ")[0].replace("__label__", "")
            dewey=dewey.replace(".","")
            if dewey_digits>0:
                if len(dewey)>dewey_digits:
                    dewey=dewey[:dewey_digits]

            if dewey in dewey_dict:
                dewey_dict[dewey].append(file)
            else:
                dewey_dict[dewey]=[file]
    training_list=[]
    test_list= []

    for key in dewey_dict.keys():
        temp_list=dewey_dict[key]
        shuffle(temp_list)
        split = max(1, math.floor(len(temp_list) * test_ratio))
        if len(temp_list)>1:
            test_list.extend(temp_list[:split])
            training_list.extend(temp_list[split:])
        else:
            training_list.extend(temp_list)

    print("Training set is  {} articles.".format(len(training_list)))
    for file in training_list:
        copyfile(os.path.join(corpus_name,file),os.path.join(training_folder,file))

    print("Test set is  {} articles.".format(len(test_list)))
    for file in test_list:
        copyfile(os.path.join(corpus_name,file),os.path.join(test_folder,file))
    print("Splitting: Complete.")
    return training_folder, test_folder


def add_wiki_to_training(wiki_corpus_name,training_folder):
    print("Adding wiki data to training set. ")
    for subdir, dirs, files in os.walk(wiki_corpus_name):
        for file in files:
            copyfile(os.path.join(wiki_corpus_name, file), os.path.join(training_folder, file))
    for subdir,dirs,files in os.walk(training_folder):
        print("New total in training set: {}".format(len(files) ))
        break
    print("Wiki data: Complete.")




def split_text(text, number_of_words_per_output_article):
    tokenized_text = word_tokenize(text, language="norwegian")
    split_count = max(1, floor(len(tokenized_text)/number_of_words_per_output_article))
    split_texts = array_split(tokenized_text,split_count)
    return split_texts

def split_training_articles(training_folder, article_length):
    training_folder_split=training_folder+"_split"

    if  os.path.exists(training_folder_split):
        shutil.rmtree(training_folder_split)
        os.makedirs(training_folder_split)
    else:
        os.makedirs(training_folder_split)

    for subdir, dirs, files in os.walk(training_folder):
        for file in files:
            f = open(os.path.join(training_folder, file), "r+")
            text=f.read()
            text= text.split(" ")
            dewey = text[0].replace("__label__", "")
            text=" ".join(text[1:])
            #print(dewey)

            texts=split_text(text,article_length)
            for i,text in enumerate(texts):

                temp = list(text)
                text=" ".join(temp)
                #print(i, text)
                f =open(os.path.join(training_folder_split,file[:-4]+"_"+str(i)+file[-4:]),"w+")
                f.write("__label__"+dewey+" "+str(text))


    return training_folder_split

def fix_corpus(folder, name_of_new_folder):

    if not os.path.exists(name_of_new_folder):
        os.makedirs(name_of_new_folder)


    arr_txt = [path for path in os.listdir(folder) if path.endswith(".txt")]
    for article_path in arr_txt:
        with open(folder+"/"+article_path,"r") as text_file:
            text = text_file.read().replace('\n',' ')
        edited_text = open(name_of_new_folder+'/'+article_path,'w')
        edited_text.write(text)
        edited_text.close()

if __name__ == '__main__':

    # fix_corpus("corpus","corpus2")
    # exit(0)
    config=load_config("default")

    #corpus=preprocess(config["name_corpus"], config["stemming"], config["stop_words"],config["sentences"], config["lower_case"], config["extra_functions"])
    wiki_corpus=preprocess_wiki(config["name_corpus"]+"_wiki", config["stemming"], config["stop_words"], config["sentences"],config["lower_case"], config["extra_functions"])
    training_folder, test_folder=split_to_training_and_test(config["name_corpus"],0.2,3)
    if config["wikipedia"]:
        add_wiki_to_training(config["name_corpus"]+"_wiki",training_folder)
    #load_config_file

    split_training_articles(training_folder,1000)

    #preprocess function
    #result is FT-format-file

    #split to training and test
    # add wiki to training
    # add fakes to training (and test?)
    #split up test and training to multiple articles


    #sum to one file

