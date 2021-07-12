# 1
# with open('news', 'r') as f:
#     text = f.read()
#     news = text.split("\n\n")
#     count = {'sport': 0, 'world': 0, "us": 0, "business": 0, "health": 0, "entertainment": 0, "sci_tech": 0}
#     for news_item in news:
#         lines = news_item.split("\n")
#         print(lines[6])
#         file_to_write = open('data/' + lines[6] + '/' + str(count[lines[6]]) + '.txt', 'w+')
#         count[lines[6]] = count[lines[6]] + 1
#         file_to_write.write(news_item)  # python will convert \n to os.linesep
#         file_to_write.close()
        
        
        
        
# #2
import pandas
import glob

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
directory_list = ["data/sport/*.txt", "data/world/*.txt","data/us/*.txt","data/business/*.txt","data/health/*.txt","data/entertainment/*.txt","data/sci_tech/*.txt",]

text_files = list(map(lambda x: glob.glob(x), directory_list))
text_files = [item for sublist in text_files for item in sublist]

training_data = []


for t in text_files:
    f = open(t, 'r')
    f = f.read()
    t = f.split('\n')
    training_data.append({'data' : t[0] + ' ' + t[1], 'flag' : category_list.index(t[6])})
    
training_data[0]        

# # #3
training_data = pandas.DataFrame(training_data, columns=['data', 'flag'])
training_data.to_csv("train_data.csv", sep=',', encoding='utf-8') # create a train data train_data1.csv is an example
print(training_data.data.shape)

# # #4
import pickle
from sklearn.feature_extraction.text import CountVectorizer


#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb")) # count_vector1.pkl is an example


# #5

from sklearn.feature_extraction.text import TfidfTransformer

#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf1.pkl","wb")) # tfidf1.pkl is an example 


# From 2 to 5 is a cycle

# #6


# # Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#clf = MultinomialNB().fit(X_train_tfidf, training_data.flag)
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)

#SAVE MODEL
pickle.dump(clf, open("nb_model.pkl", "wb"))


# #7



import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

docs_new = "Messi joins other football team"  # this is an input for testing whether the model is working or not.
docs_new = [docs_new]

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("nb_model.pkl","rb"))  # Firat algorithm loaded

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

print("Baysian Prediction" + "  " + category_list[predicted[0]])

# plt.hist(predicted, facecolor='peru',edgecolor='white', bins=10)
# plt.show()
# #8

import numpy as np
import matplotlib.pyplot as plt
import random
predicted = loaded_model.predict(X_test)
result_bayes = pandas.DataFrame( {'actual_labels': y_test,'predicted_labels': predicted})
result_bayes.to_csv('res_bayes.csv', sep = ',')

#roc r and d
#roc calculation and plot
import pandas
import numpy as np
import matplotlib. pyplot as plt
from sklearn.metrics import confusion_matrix

rocp = []

cnf_matrix = confusion_matrix(y_test, predicted)

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)

# Fall out or false positive rate
FPR = FP/(FP+TN)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

rocp.append([TPR,FPR])
print("Rocp array:",rocp)
print('TPR')
print(TPR)
print('FPR')
print(FPR)  
print("ACC")
print(ACC)  

plt.scatter(TPR, FPR)
plt.xlabel('True positive Rate')
plt.ylabel('False positive Rate')
plt.plot([0,1])
#plt.show()             
        
        
                


# for predicted_item, result in zip(predicted, y_test):
#     print(category_list[predicted_item], ' - ', category_list[result])
    
    
# #9


# from sklearn.metrics import confusion_matrix  

# confusion_mat = confusion_matrix(y_test,predicted)
# print(confusion_mat)


# #10


##################################### Nural  Model End #################################
from sklearn.neural_network import MLPClassifier

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

clf_neural.fit(X_train, y_train)




# #11


pickle.dump(clf_neural, open("softmax.pkl", "wb"))

# #12


predicted = clf_neural.predict(X_test)
result_softmax = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_softmax.to_csv('res_softmax.csv', sep = ',')

print("Nural Network Prediction" + "  " + category_list[predicted[0]])
# for predicted_item, result in zip(predicted, y_test):
#     print(category_list[predicted_item], ' - ', category_list[result])

##################################### Nural Model End #################################    
    
    
    ##################################### SVM Model End #################################
# #13


from sklearn import svm
clf_svm = svm.LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_svm.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_svm, open("svm.pkl", "wb"))


#14


predicted = clf_svm.predict(X_test)
result_svm = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_svm.to_csv('res_svm.csv', sep = ',')
print( "SVM Prediction" + "  " + category_list[predicted[0]])
# for predicted_item, result in zip(predicted, y_test):
#     print(category_list[predicted_item], ' - ', category_list[result])
##################################### SVM Model End #################################    
  
  
#15  
##################################### Decision Model Start #################################
# We  applied The  Decision Tree Algorithm 
#Total Lines 09
from sklearn.tree import DecisionTreeClassifier
clf_decision= DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_decision.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_decision, open("decisionmodel.pkl", "wb"))



#Train the model using the training sets y_pred=clf.predict(X_test)


predicted_decision = clf_decision.predict(X_test)
result_decision = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted_decision})
result_decision.to_csv('res_decision.csv', sep = ',')
print( "Decision Tree Prediction" + "  "  + category_list[predicted_decision[0]])

##################################### Decision Model End #################################


# #15
# We will build a model using Random Forest Algorithm

##################################### Random Model Start #################################
#16 steps Total Line 09
#Implementing The  Random Forest Model
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score
import matplotlib.pyplot as plt

#Create a Gaussian Classifier n_estimators=100
clf_random=RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_random.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_random, open("random.pkl", "wb"))



#Train the model using the training sets y_pred=clf.predict(X_test)


random_pred=clf_random.predict(X_test)
result_random = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': random_pred})
result_random.to_csv('res_random.csv', sep = ',')
print("Random Forest Prediction" + "  " + category_list[random_pred[0]])
fscore = f1_score(y_test, random_pred, average='macro')
# The below code is for showing polt
plt.hist(fscore, facecolor='peru',edgecolor='white', bins=10)
plt.show()


##################################### Random Model End #################################


#7 Adaboost Alg
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,precision_score,recall_score

#For plotting 
import matplotlib.pyplot as plt


classifier_adaboost = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200
)            
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

classifier_adaboost.fit(X_train_tfidf, training_data.flag)
pickle.dump(classifier_adaboost, open("adaboost.pkl", "wb"))

adaboost_pred=clf_random.predict(X_test)
result_adaboost = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': adaboost_pred})
result_adaboost.to_csv('res_adaboost.csv', sep = ',')
print("Adaboost Prediction" + "  " + category_list[adaboost_pred[0]])
fscore = f1_score(y_test, adaboost_pred, average='macro')
print('Fscore Value: %.3f' % fscore)
# calculate prediction
precision = precision_score(y_test, adaboost_pred, average='weighted')
print('Precision: %.3f' % precision)
recallscore = recall_score(y_test, adaboost_pred, average='weighted')
print(('RecallScore: %.3f' % recallscore))


## ROC



# method I: plt


##

# plt.hist(fscore, facecolor='peru',edgecolor='white', bins=10)
# plt.show()

#Import scikit-learn metrics module for accuracy calculation
#from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#clf.predict([[3, 5, 4, 2]])
#Adaboost classifoer

#for model comparison rutba
#Import scikit-learn metrics module for confusion matrix, accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#For plotting 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score
# fscore_ada = f1_score(y_test, ada_pred, average='macro')
# print(fscore_ada)
# plt.hist(fscore_ada, facecolor='peru',edgecolor='white', bins=10)
# plt.show()
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score
import random as random 


scores = []
fscorearr = []
precarr = []
recallarr = []
accuarr = []
ranks = []
#GET VECTOR COUNT
count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(training_data.data)


#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.3, random_state=42)

names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM",
         "Gradient_Boosting", "Decision_Tree", "Extra_Trees", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA", "SGD"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=100),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    # SGDClassifier(loss="hinge", penalty="l2")
     ]

 
for name, clf in zip(names, classifiers):
    
    model = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    predicted = model.predict(X_test)
    
    scores.append(score)
    plt.hist(score, facecolor='peru',edgecolor='white', bins=10)
    
    r = random.random()
    g = random.random()
    b = random.random()
    my_color = (r,g,b)
    fscore = f1_score(y_test, predicted, average='macro')
    precision = precision_score(y_test, predicted, average='micro')
    recall = recall_score(y_test, predicted, average='macro')
    accuracy = accuracy_score(y_test, predicted)
    conf_matrix = confusion_matrix(y_test, predicted)
    plt.hist(fscore, label=name, facecolor=my_color,edgecolor='white', bins=10)
    
    #Print Confusion matrix, accuracy, f-score, precision and recall values per classifier
    print("Precision:")
    print("F-scores:"+str(fscore))
    print(name+": "+str(accuracy))
    print(name)
    print(conf_matrix)
    print("Recall:"+str(recall))


    # Create fscore array and store all the f-score values
    fscorearr.append(fscore*100)
    fscorearr.sort()

    # Create precision array and store all the precision values
    precarr.append(precision*100)
    precarr.sort()

    # Create recall array and store all the recall values
    recallarr.append(recall*100)
    recallarr.sort()

    accuarr.append(accuracy*100)
    accuarr.sort()

for fscore in fscorearr:        
    ranks.append(fscorearr.index(fscore) + 1)
    r = random.random()
    g = random.random()
    b = random.random()
    my_color = (r,g,b)

# Plotting accuracy per Classifier  
x_axis = accuarr
y_axis = ['Linear_SVM', 'Poly_SVM', 'Neural_Net', 'AdaBoost', 'Decision_Tree', 'Random_Forest', 'RBF_SVM', 'Gradient_Boosting', 'Extra_Trees', 'Nearest_Neighbors']

plt.barh(y_axis,x_axis,facecolor=my_color,edgecolor='white')
ax1 = plt.subplot(111)
ret = ax1.barh(y_axis,x_axis,facecolor=my_color,edgecolor='white')
ret[0].set_color('#40ff00')
ret[1].set_color('#00ffff')
ret[2].set_color('#0000ff')
ret[3].set_color('#4000ff')
ret[4].set_color('#bf00ff')
ret[5].set_color('#ff0080')
ret[6].set_color('#0080ff')
ret[7].set_color('#ff00bf')
ret[8].set_color('#ffff00')
ret[9].set_color('#00ff80')
plt.title('Accuracy per Classifier')
plt.ylabel('Classifiers')
plt.xlabel('Accuracy')
print(accuarr)
plt.show()

# Plotting f-score per Classifier  
x_axis = fscorearr
y_axis = ['Linear_SVM', 'Poly_SVM', 'Neural_Net', 'AdaBoost', 'Decision_Tree', 'Random_Forest', 'RBF_SVM', 'Gradient_Boosting', 'Extra_Trees', 'Nearest_Neighbors']

plt.barh(y_axis,x_axis,facecolor=my_color,edgecolor='white')
plt.title('F-score per Classifier')
plt.ylabel('Classifiers')
plt.xlabel('F-score')
print(fscorearr)
print("Ranks: "+str(ranks))
plt.show()

# Plotting recall per Classifier  
x_axis = recallarr
y_axis = ['Linear_SVM', 'Poly_SVM', 'Neural_Net', 'AdaBoost', 'Decision_Tree', 'Random_Forest', 'RBF_SVM', 'Gradient_Boosting', 'Extra_Trees', 'Nearest_Neighbors']

plt.barh(y_axis,x_axis,facecolor=my_color,edgecolor='white')
plt.title('Recall per Classifier')
plt.ylabel('Classifiers')
plt.xlabel('Recall')
print(recallarr)
plt.show()


# Plotting precision per Classifier  
x_axis = precarr
y_axis = ['Linear_SVM', 'Poly_SVM', 'Neural_Net', 'AdaBoost', 'Decision_Tree', 'Random_Forest', 'Gradient_Boosting', 'Extra_Trees', 'Nearest_Neighbors', 'RBF_SVM']

plt.barh(y_axis,x_axis,facecolor=my_color,edgecolor='white')
plt.title('Precision per Classifier')
plt.ylabel('Classifiers')
plt.xlabel('Precision')
print(precarr)
plt.show()




# plt.bar(score, facecolor='peru',edgecolor='white', bins=10)
# print(score)
    
#Roc
