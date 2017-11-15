import fasttext
import time
import os
from datetime import datetime
from evaluator import evaluate_fasttext

def create_classifier(training_file,name,epoch,lr1,DIMENSIONS, up_rate,ws,loss1,wiki,save_location):
    #print(epoch,lr1,DIMENSIONS,ws,up_rate,loss1,wiki)
    if wiki:
        return fasttext.supervised(training_file,
                            os.path.join(save_location,name ),

                               epoch=epoch,
                               lr=lr1,
                                pretrained_vectors="wiki.no.vec",
                               dim=DIMENSIONS,
                               ws=ws,
                               lr_update_rate=up_rate,
                               loss=loss1,
                                   bucket=5000000)
    else:

        return fasttext.supervised(training_file,
                            os.path.join(save_location,name ),

                               epoch=epoch,
                               lr=lr1,
                               dim=DIMENSIONS,
                               ws=ws,
                               lr_update_rate=up_rate,
                               loss=loss1,
                                   bucket=10)

#parameters

def train_fasttext_models(train_file,test_file,save_location, epoch_vector,learning_rate_vector,learning_rate_update_vector,word_window_vector,loss_vector,wiki_vector,fasttext_k, minimum_articles,save_model):
    temp_file=open(os.path.join(save_location,"training_file.txt"),"w")
    temp_file.write(train_file)
    temp_file = open(os.path.join(save_location, "test_file.txt"), "w")
    temp_file.write(test_file)

    if isinstance(epoch_vector,str):
        epoch_vector=[int(epoch_vector)]
    else:
        epoch_vector= list(map(int, epoch_vector))

    EPOCHs=epoch_vector

    if isinstance(learning_rate_vector,str):
        learning_rate_vector=[float(learning_rate_vector)]
    else:
        learning_rate_vector= list(map(float, learning_rate_vector))
    LRs=learning_rate_vector

    if isinstance(learning_rate_update_vector,str):
        learning_rate_update_vector=[float(learning_rate_update_vector)]
    else:
        learning_rate_update_vector= list(map(float, learning_rate_update_vector))
    UP_RATEs=learning_rate_update_vector


    if isinstance( word_window_vector,str):
        word_window_vector=[int( word_window_vector)]
    else:
        word_window_vector= list(map(int,  word_window_vector))
    WS=word_window_vector



    if isinstance(loss_vector,str):
        loss_vector=[loss_vector]
    LOSS=loss_vector

    if isinstance( wiki_vector,str):
        wiki_vector=[ wiki_vector]
    WIKI_VEC=wiki_vector

    if isinstance(fasttext_k,str):
        fasttext_k=[int(fasttext_k)]
    else:
        fasttext_k= list(map(int, fasttext_k))
    Ks=fasttext_k

    DIMENSIONS = 300
    tid = time.time()
    total_iterations = len(EPOCHs) * len(LRs) * len(UP_RATEs) * len(WS) * len(LOSS) * len(WIKI_VEC)
    count = 0
    for epoch in EPOCHs:
        for lr in LRs:
            for up_rate in UP_RATEs:
                for ws in WS:
                    for loss in LOSS:
                        for wiki_vec in WIKI_VEC:
                            count += 1
                            print("Starting to train FastText models ")
                            print("It is : " + str(datetime.now()))
                            tid = time.time()
                            print("Iteration nr {} out of {}".format(count, total_iterations))
                            print("Creating and training classifier:")
                            name="model-{}-{}-{}-{}-{}-{}".format(str(epoch),str(lr).replace(".",""),str(up_rate),str(ws),loss,str(wiki_vec))
                            if save_model:
                                classifier = create_classifier(os.path.join(save_location,"training_file.txt"),name, epoch, lr, DIMENSIONS, up_rate, ws, loss, wiki_vec,save_location)
                            else:
                                classifier = create_classifier(os.path.join(save_location,"training_file.txt"),"temp", epoch, lr, DIMENSIONS, up_rate, ws, loss, wiki_vec,save_location)
                            print("Creating the classifier took {} seconds.".format(time.time() - tid))
                            tid = time.time()
                            if not os.path.exists(os.path.join(save_location,"logs")):
                                os.makedirs(os.path.join((save_location,"logs"))
                            with open(os.path.join(os.path.join(save_location,"logs"),"log-"+name+".txt"),"w") as logfile:

                                logfile.write("epoch:::{}\n".format(str(epoch)))
                                logfile.write("lr:::{}\n".format(str(lr)))
                                logfile.write("lr_up_rate:::{}\n".format(str(up_rate)))
                                logfile.write("Word context length:::{}\n".format(str(ws)))
                                logfile.write("loss:::{}\n".format(loss))
                                logfile.write("wiki_vec:::{}\n".format(str(wiki_vec)))
                                for k in Ks:
                                    print("K-run:{}".format(k))
                                    result = classifier.test(os.path.join(save_location, "test_file.txt"), k)
                                    print("Running took {} seconds.".format(time.time() - tid))
                                    tid = time.time()
                                    precision = result.precision
                                    recall = result.recall
                                    logfile.write("log k={}:::{}:::{}\n\n".format(k, precision, recall))
                                logfile.write(evaluate_fasttext(os.path.join(save_location, "test_file.txt"), classifier))

