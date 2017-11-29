import operator
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
import operator





def evaluate_predictions(correct_list, prediction_list):
    c = confusion_matrix(correct_list, prediction_list)
    true_positives_vector = np.diag(c)
    precision_vector = true_positives_vector / np.sum(c, axis=0, dtype=np.float)
    recall_vector = np.diag(c) / np.sum(c, axis=1, dtype=np.float)
    # print(precision_vector)
    # print(recall_vector)

    true_positives_vector = []
    false_positives_vector = []
    false_negatives_vector = []
    true_negatives_vector = []
    for i in range(0, len(c)):
        TP_of_class_i = c[i, i]
        FP_of_class_i = np.sum(c, axis=0)[i] - c[i, i]  # The corresponding column for class_i - TP
        FN_of_class_i = np.sum(c, axis=1)[i] - c[i, i]  # The corresponding row for class_i - TP
        TN_of_class_i = np.sum(c) - TP_of_class_i - FP_of_class_i - FN_of_class_i
        true_positives_vector.append(TP_of_class_i)
        false_positives_vector.append(FP_of_class_i)
        false_negatives_vector.append(FN_of_class_i)
        true_negatives_vector.append(TN_of_class_i)
    # print (true_positives_vector)
    # print(false_negatives_vector)
    # print(false_positives_vector)
    # print(true_negatives_vector)

    #print(prediction_list)
    correct_guesses={}
    mistaken_guesses={}
    total_articles={}
    for dewey in correct_list:
        if dewey not in total_articles:
            total_articles[dewey]=1
            mistaken_guesses[dewey]=0
            correct_guesses[dewey]=0
        else:
            total_articles[dewey] += 1

    for correct,predict in zip(correct_list,prediction_list):
        #print("Correct: {} , predict: {}".format(correct,predict))
        if correct==predict:
            correct_guesses[correct]+=1
        else:
            if predict not in mistaken_guesses:

                mistaken_guesses[predict]=1
                total_articles[predict]=0
                correct_guesses[predict]=0
            else:
                mistaken_guesses[predict] += 1



    sorted_total = sorted(total_articles.items(), key=operator.itemgetter(1), reverse=True)
    return_text=""
    total=len(prediction_list)
    print("Length of predictions is: {}".format(total))
    total_correct=0
    dewey_nr=[]
    totals=[]
    corrects=[]
    incorrects=[]
    mistakens=[]
    accuracies=[]
    for dewey in sorted_total:
        dewey=dewey[0]
        accuracy=0
        if float(total_articles[dewey])>0:
            accuracy=correct_guesses[dewey]/float(total_articles[dewey])
        confidence=0
        if correct_guesses[dewey]+mistaken_guesses[dewey]>0:
            confidence=correct_guesses[dewey]/float(correct_guesses[dewey]+mistaken_guesses[dewey])
        total_correct+=correct_guesses[dewey]
        return_text+=("Dewey: {},total: {}, correct: {}, incorrect: {}, mistaken: {}, accuracy: {:.3f}, confidence: {:.4f},".format(dewey,total_articles[dewey],correct_guesses[dewey],total_articles[dewey]-correct_guesses[dewey],mistaken_guesses[dewey],accuracy,confidence))+"\n"

    return return_text,total,total_correct


def calculate_f1(correct_list,prediction_lists):
    prediction_lists2=[]
    for i in range(len(prediction_lists[0])):
        prediction_lists2.append([])
    for prediction_list in prediction_lists:
        for k in range(len(prediction_list[0])):
            prediction_lists2[k].append(prediction_list[k])
    prediction_lists=prediction_lists2

    total_correct=0
    total_text=[]
    text,total,correct=evaluate_predictions(correct_list,prediction_lists[0])
    accuracy=correct/float(total)

    for prediction_list in prediction_lists:
        tmp,total,correct=evaluate_predictions(correct_list,prediction_list)
        total_text.append(tmp)
        total_correct+=correct

    recall=total_correct/float(total)
    print("Recall: {}, Accuracy: {}".format(recall,accuracy))
    f1_score= 2*(accuracy*recall)/float(accuracy+recall)
    return f1_score, total_text


def majority_rule(predictions, k):
    #returner liste med de k mest populære prediksjonene.

    prediction_lists2 = []
    for i in range(len(predictions[0])):
        prediction_lists2.append([])
    for prediction_list in predictions:
        for j in range(len(prediction_list[0])):
            prediction_lists2[j].append(prediction_list[j])
    prediction_lists=prediction_lists2
    print(prediction_lists)
    weights = [1/n for n in range(1,k+2)]

    pred_dict = dict()
    for n in range(0, len(prediction_lists)):
        top_k_preds = [pred for pred in Counter(prediction_lists[n]).most_common()]
        for prediction in top_k_preds:
            freq = prediction[1]
            dewey = prediction[0]
            if not dewey in pred_dict :
                pred_dict[dewey] = freq*weights[n]
            else:
                pred_dict[dewey]+=freq*weights[n]
    sorted_tuples = sorted(pred_dict.items(), key=operator.itemgetter(1), reverse=True)
    top_k_preds = []
    for i in range(0,k):
        top_k_preds.append(sorted_tuples[i][0])

    return top_k_preds


if __name__ == '__main__':
    preds = [["123", "321", "777"],["123", "323", "777"],["999","888","777"]]

    majority_rule(preds,5)

