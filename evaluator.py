import fasttext
import operator
from evaluator2 import calculate_f1


def evaluate_fasttext(test_file,classifier,top_k_labels):
    return_text=""
    texts=[]
    correct=[]
    #test_file_name="corpus_test2-2.txt"
    with open(test_file, "r")as f:
        for line in f.readlines():
            texts.append(line)
            correct.append(line.split(" ")[0].replace("__label__",""))
    guesses = classifier.predict(texts, k=top_k_labels) #labels
    # guesses=[]
    # for label in labels:
    #
    #     guesses.append(label[0][0])
    return calculate_f1(correct,guesses)


    accuracy = {}
    for i, article in enumerate(texts):
        label = article.partition(' ')[0].replace("__label__","")
        guess = labels[i][0][0]
        # print (label)
        # print (guess)
        if label in accuracy:
            accuracy[label][0] += 1
            if guess == label:
                accuracy[label][1] += 1
            else:
                if guess in accuracy:
                    accuracy[guess][2] += 1
                else:
                    accuracy[guess] = [0, 0, 1]
        else:
            accuracy[label] = [1, 0, 0]
            if guess == label:
                accuracy[label][1] += 1
            else:
                if guess in accuracy:
                    accuracy[guess][2] += 1
                else:
                    accuracy[guess] = [0, 0, 1]

    for i in range(min(len(texts), len(labels))):

        temp = texts[i][:16].split(" ")
        # print(temp[0], labels[i])
        if temp[0] != labels[i][0][0]:
            # print("Feil:")
            #print(temp[0], labels[i])
            pass
            # if len(texts[i].split(" "))<250:
            #
            #     print(texts[i])
    sorted_acc = sorted(accuracy.items(), key=operator.itemgetter(1), reverse=True)
    confidence_sum = 0
    riktige_sum = 0
    gale_sum = 0
    for article in sorted_acc:
        # print (article)
        # print("{} , Accuracy: {:.3f}  Artikler: {}  Riktige: {} Feil: {}".format(key,(accuracy[key][1]/accuracy[key][0]),accuracy[key][0],accuracy[key][1],accuracy[key][2]))
        guesses=float(article[1][2] + article[1][1])
        if article[1][0] >= 1:
            if guesses==0:
                confidence=0
            else:
                confidence=float(article[1][1] / guesses)
            confidence_sum += confidence
            riktige_sum += article[1][1]
            accuracy_temp=(article[1][1] / article[1][0])
            wrong_guess=article[1][0] - article[1][1]
            gale_sum += article[1][2]
            #print("Dewey nr: {} -  Accuracy: {:.3f}  Artikler: {}  Riktige: {} Feil: {} Feil gjetning her: {} Confidence:{:.4f} \n ".format(article[0].replace("__label__", ""),(article[1][1] / article[1][0]),article[1][0], article[1][1],article[1][0] - article[1][1],article[1][2], float(article[1][1] / float(article[1][2] + article[1][1]))))
            # print("{} , Accuracy: {:.3f}  Artikler: {}  Riktige: {} Feil: {}".format(key,(accuracy[key][1]/accuracy[key][0]),accuracy[key][0],accuracy[key][1],accuracy[key][0]-accuracy[key][1]))
            return_text+=("Dewey nr: {} -  Accuracy: {:.3f}  Artikler: {}  Riktige: {} Feil: {} Feil gjetning her: {} Confidence:{:.4f} \n ".format(article[0].replace("__label__", ""),accuracy_temp,article[1][0], article[1][1],wrong_guess,article[1][2],confidence ))
    result = classifier.test(test_file, 1)
    return_text+=("Average confidence: {} \n".format(confidence_sum / float(len(sorted_acc))))
    print("Average confidence: {} \n".format(confidence_sum / float(len(sorted_acc))))
    return_text+=("Samlet confidence: {}\n".format(riktige_sum / float(gale_sum + riktige_sum)))
    return_text+=("Overall accuracy: {} \n".format(result.precision))
    return_text+=("Overall recall: {} \n".format(result.recall))
    print("Samlet confidence: {}\n".format(riktige_sum / float(gale_sum + riktige_sum)))
    print("Overall accuracy: {} \n".format(result.precision))
    print("Overall recall: {} \n".format(result.recall))

    return return_text