import operator

def evaluate_predictions(correct_list, prediction_list):
    #print(prediction_list)
    correct_guesses={}
    mistaken_guesses={}
    total_guesses={}
    for dewey in correct_list:
        if dewey not in total_guesses:
            total_guesses[dewey]=1
            mistaken_guesses[dewey]=0
            correct_guesses[dewey]=0
        else:
            total_guesses[dewey] += 1

    for correct,predict in zip(correct_list,prediction_list):
        if correct==predict:
            correct_guesses[correct]+=1
        else:
            mistaken_guesses[predict]+=1



    sorted_total = sorted(total_guesses.items(), key=operator.itemgetter(1), reverse=True)
    return_text=""
    total=len(prediction_list)
    total_correct=0
    dewey_nr=[]
    totals=[]
    corrects=[]
    incorrects=[]
    mistakens=[]
    accuracies=[]
    for dewey in sorted_total:
        dewey=dewey[0]
        accuracy=correct_guesses[dewey]/float(total_guesses[dewey])
        confidence=0
        if correct_guesses[dewey]+mistaken_guesses[dewey]>0:
            confidence=correct_guesses[dewey]/float(correct_guesses[dewey]+mistaken_guesses[dewey])
        total_correct+=correct_guesses[dewey]
        return_text+=("Dewey: {},total: {}, correct: {}, incorrect: {}, mistaken: {}, accuracy: {:.3f}, confidence: {:.4f},".format(dewey,total_guesses[dewey],correct_guesses[dewey],total_guesses[dewey]-correct_guesses[dewey],mistaken_guesses[dewey],accuracy,confidence))+"\n"

    return return_text,total,total_correct


def calculate_f1(correct_list,prediction_lists):
    prediction_lists2=[[]]*len(prediction_lists[0])
    for prediction_list in prediction_lists:
        for k in range(len(prediction_list)):
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







