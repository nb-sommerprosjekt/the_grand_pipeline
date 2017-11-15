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
from fast_text import train_fasttext_models
import sys

def load_config(config_file):
    print("Loading config file: {}".format(config_file))
    config={}

    with open("config/"+config_file+".txt","r") as file:
        for line in file.readlines():
            if len(line)>4:
                if line[0]!="#":
                    temp=line.split(":::")
                    temp[1]=temp[1].strip()
                    if temp[1] is int:
                        config[temp[0]]=int(temp[1])
                    else:
                        if len(temp[1].split(","))>1:
                            config[temp[0]]=temp[1].split(",")
                        else:
                            config[temp[0]] = temp[1]
    print("Config-file loaded")
    return config

def preprocess(corpus_name,stemming, stop_words,sentences,lower_case,extra_functions):
    counter=0
    rootdir="tgc"
    corpus_name_folder=os.path.join("data_set",corpus_name+"_folder")
    corpus_name_location=os.path.join(corpus_name_folder,corpus_name)

    if not os.path.exists(corpus_name_folder):
        os.makedirs(corpus_name_folder)
    if not os.path.exists(corpus_name_location):
        os.makedirs(corpus_name_location)
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
                    file=open(os.path.join(corpus_name_location, file[5:]),"w")
                    file.write("__label__"+dewey+" "+text)
        print("Preprocessed corpus saved in the folder {}".format(corpus_name_location))
        return corpus_name_folder  #Kanskje kjøre neste steg.



    else:
        print("Corpus is already created, using {} ".format(corpus_name_location))
        return corpus_name_folder #Kanskje kjøre neste steg.

def preprocess_wiki(corpus_folder,wiki_corpus_name,stemming, stop_words, sentences,lower_case, extra_functions):

    rootdir="tgc_wiki"
    counter=0
    wiki_corpus_folder=os.path.join(corpus_folder,wiki_corpus_name)
    if not os.path.exists(wiki_corpus_folder):
        os.makedirs(wiki_corpus_folder)
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

                f2 = open(os.path.join(wiki_corpus_folder, file), "w+")
                f2.write("__label__" + dewey + " " + text)
        print("Preprocessed  wiki_corpus saved in the folder {}".format(wiki_corpus_name))
        return wiki_corpus_folder  #Kanskje kjøre neste steg.

    else:
        print("Wiki corpus is already created, using {} ".format(wiki_corpus_name))
        return wiki_corpus_folder  #Kanskje kjøre neste steg.



def split_to_training_and_test(corpus_folder,corpus_name, test_ratio,dewey_digits):
    print("Splitting to training and test:")
    training_folder= os.path.join(corpus_folder,corpus_name+"_training")
    test_folder= os.path.join(corpus_folder,corpus_name+"_test")
    corpus_name_location=os.path.join(corpus_folder ,corpus_name)

    dewey_dict={}

    for subdir, dirs, files in os.walk(corpus_name_location):
        for file in files:
            f = open(os.path.join(corpus_name_location, file), "r+")
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


    if  os.path.exists(training_folder):
        print("The test-folder already exists. No action will be taken. ")
        print("Training set is  {} articles.".format(len(training_list)))
        # shutil.rmtree(training_folder)
        # os.makedirs(training_folder)
    else:
        os.makedirs(training_folder)
        print("Training set is  {} articles.".format(len(training_list)))
        for file in training_list:
            copyfile(os.path.join(corpus_name_location, file), os.path.join(training_folder, file))

    if  os.path.exists(test_folder):
        print("The test-folder already exists. No action will be taken. ")
        print("Test set is  {} articles.".format(len(test_list)))
        # shutil.rmtree(test_folder)
        # os.makedirs(test_folder)
    else:
        os.makedirs(test_folder)
        print("Test set is  {} articles.".format(len(test_list)))
        for file in test_list:
            copyfile(os.path.join(corpus_name_location, file), os.path.join(test_folder, file))






    print("Splitting: Complete.")
    return training_folder, test_folder


def add_wiki_to_training(wiki_corpus_folder,training_folder):
    print("Adding wiki data to training set. ")
    for subdir, dirs, files in os.walk(wiki_corpus_folder):
        for file in files:
            copyfile(os.path.join(wiki_corpus_folder, file), os.path.join(training_folder, file))
    for subdir,dirs,files in os.walk(training_folder):
        print("New total in training set: {}".format(len(files) ))
        break
    print("Wiki data: Complete.")




def split_text(text, number_of_words_per_output_article):
    tokenized_text = word_tokenize(text, language="norwegian")
    split_count = max(1, floor(len(tokenized_text)/int(number_of_words_per_output_article)))
    split_texts = array_split(tokenized_text,split_count)
    return split_texts

def split_articles(folder, article_length):
    article_length=int(article_length)
    folder_split=folder+"_split"

    if  os.path.exists(folder_split):
        # shutil.rmtree(training_folder_split)
        # os.makedirs(training_folder_split)
        print("The split-folder already exists. No action will be taken. ")
        return folder_split
    else:
        os.makedirs(folder_split)

    for subdir, dirs, files in os.walk(folder):
        for file in files:
            with  open(os.path.join(folder, file), "r+")as f:
                text=f.read()
            text= text.split(" ")
            dewey = text[0].replace("__label__", "")
            text=" ".join(text[1:])
            #print(dewey)

            texts=split_text(text,article_length)
            for i,text in enumerate(texts):

                temp = list(text)
                text=" ".join(temp)
                #print(i, len(text))
                #print(os.path.join(folder_split,file[:-4]+"_"+str(i)+file[-4:]))
                with open(os.path.join(folder_split,file[:-4]+"_"+str(i)+file[-4:]),"w+") as f:
                    f.write("__label__"+dewey+" "+str(text))


    return folder_split


def remove_unecessary_articles(training_folder_split,corpus_folder,minimum_articles,dewey_digits):
    dewey_digits=int(dewey_digits)
    rubbish_folder=os.path.join(corpus_folder,"rubbish")
    if  os.path.exists(rubbish_folder):
        # shutil.rmtree(training_folder_split)
        # os.makedirs(training_folder_split)
        print("The rubbish-folder already exists. No action will be taken. ")
    else:
        print("Creating rubbish-folder")
        os.makedirs(rubbish_folder)

    dewey_dict = {}


    for subdir, dirs, files in os.walk(training_folder_split):
        for file in files:
            with open(os.path.join(training_folder_split, file), "r+") as f:
                f.seek(0)
                text = f.read()
                f.close()

            dewey = text.split(" ")[0].replace("__label__", "")
            dewey = dewey.replace(".", "")
            if int(dewey_digits) > 0:
                if len(dewey) > int(dewey_digits):
                    dewey = dewey[:dewey_digits]

            if dewey in dewey_dict:
                dewey_dict[dewey].append(file)
            else:
                dewey_dict[dewey] = [file]
    valid_deweys=set()
    for key in dewey_dict.keys():
        if len(dewey_dict[key])<int(minimum_articles):
            for file in dewey_dict[key]:
                os.rename(os.path.join(training_folder_split,file),os.path.join(rubbish_folder,file))
        else:
            for file in dewey_dict[key]:
                valid_deweys.add(key)
                with open(os.path.join(training_folder_split, file), "r+") as f2:
                    f2.seek(0)
                    text = f2.read()
                    temp1=len(text)
                    text = text.split(" ")
                    dewey = text[0].replace("__label__", "")
                    text = " ".join(text[1:])
                    temp2=len(text)

                    if len(dewey) > dewey_digits:
                        dewey = dewey[:dewey_digits]
                    with open(os.path.join(training_folder_split, file), "r+") as f3:
                        f3.seek(0)
                        f3.write("__label__" + dewey + " " + str(text))
                        f3.truncate()
                        f3.close()

    print("Removed unnecessary dewey numbers. There are {} unique dewey numbers in the training set.".format(len(valid_deweys)))
    return rubbish_folder,valid_deweys


def prep_test_set(test_folder,valid_deweys,article_length,dewey_digits):
    print("Prepping test-folder.")
    dewey_digits=int(dewey_digits)
    test_folder_split=split_articles(test_folder,article_length)

    print("Test-folder split.")

    for subdir, dirs, files in os.walk(test_folder_split):
        for file in files:
            #print(len(files))
            with open(os.path.join(test_folder_split, file), "r+") as f:
                text = f.read()
                text=text.split(" ")
                dewey = text[0].replace("__label__", "")
                text=" ".join(text[1:])
                dewey = dewey.replace(".", "")
                if len(dewey) > int(dewey_digits):
                    dewey = dewey[:dewey_digits]
                    #print(dewey)
                #print(dewey)
                if dewey in valid_deweys:

                    #print("Not thrash")
                    f.seek(0)
                    f.write("__label__" + dewey + " " + str(text))
                    f.truncate()
                else:
                    os.rename(os.path.join(test_folder_split, file), os.path.join(rubbish_folder, file))
    return test_folder_split





def load_set(folder):
    print("Loading {} now".format(folder))
    total_tekst = ""
    counter = 0

    for subdir, dirs, files in os.walk(folder):
        for file in files:

            if counter % 1000 == 0:
                print("Done {} out of {}".format(counter, len(files)))
            counter += 1
            with open(os.path.join(folder, file), "r+") as f:
                f.seek(0)
                text = f.read()
            total_tekst +=  text + '\n'
    return total_tekst

def create_folder(path):
    os.makedirs(path, exist_ok=True)



# def fix_corpus(folder, name_of_new_folder):
#
#     if not os.path.exists(name_of_new_folder):
#         os.makedirs(name_of_new_folder)
#
#
#     arr_txt = [path for path in os.listdir(folder) if path.endswith(".txt")]
#     for article_path in arr_txt:
#         with open(folder+"/"+article_path,"r") as text_file:
#             text = text_file.read().replace('\n',' ')
#         edited_text = open(name_of_new_folder+'/'+article_path,'w')
#         edited_text.write(text)
#         edited_text.close()

if __name__ == '__main__':

    # fix_corpus("corpus","corpus2")
    # exit(0)
    print("Trying to load config-file named: {}".format(sys.argv[1]))
    config=load_config(sys.argv[1])

    corpus_folder=preprocess(config["name_corpus"], config["stemming"], config["stop_words"],config["sentences"], config["lower_case"], config["extra_functions"])
    wiki_corpus_folder=preprocess_wiki(corpus_folder,config["name_corpus"]+"_wiki", config["stemming"], config["stop_words"], config["sentences"],config["lower_case"], config["extra_functions"])
    training_folder, test_folder=split_to_training_and_test(corpus_folder, config["name_corpus"],0.2,3)
    if config["wikipedia"]:
        wiki_corpus_folder = preprocess_wiki(corpus_folder, config["name_corpus"] + "_wiki", config["stemming"],
                                             config["stop_words"], config["sentences"], config["lower_case"],
                                             config["extra_functions"])

        add_wiki_to_training(wiki_corpus_folder,training_folder)
    #load_config_file

    training_folder_split=split_articles(training_folder,1000)

    rubbish_folder,valid_deweys=remove_unecessary_articles(training_folder_split,corpus_folder,config["minimum_articles"],config["dewey_digits"])
    #print(valid_deweys)
    test_folder_split=prep_test_set(test_folder,valid_deweys,config["article_size"],config["dewey_digits"])

    test_text=load_set(test_folder_split)
    training_text=load_set(training_folder_split)

    create_folder(os.path.join("fasttext",config["fasttext_run_name"]))
    train_fasttext_models(training_text,test_text,os.path.join("fasttext",config["fasttext_run_name"]),config["epochs"],config["lr"],config["lr_update"],config["word_window"],config["loss"],config["wiki_vec"],config["fasttext_k"],config["minimum_articles"],config["save_model"])


    #preprocess function
    #result is FT-format-file

    #split to training and test
    # add wiki to training
    # add fakes to training (and test?)
    #split up test and training to multiple articles


    #sum to one file

