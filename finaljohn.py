# with open('news', 'r') as f:
#     text = f.read()
#     news = text.split("\n\n")
#     count = {'sport': 0, 'world': 0, "us": 0, "business": 0, "health": 0, "entertainment": 0, "sci_tech": 0}
#     for news_item in news:
#         lines = news_item.split("\n")
#        # print(lines[6])
#         file_to_write = open('data/' + lines[6] + '/' + str(count[lines[6]]) + '.txt', 'w+')
#         count[lines[6]] = count[lines[6]] + 1
#         file_to_write.write(news_item)  # python will convert \n to os.linesep
#         file_to_write.close()
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
training_data = pandas.DataFrame(training_data, columns=['data', 'flag'])
training_data.to_csv("train_data.csv", sep=',', encoding='utf-8')
print(training_data.data.shape)
import pickle
from sklearn.feature_extraction.text import CountVectorizer


#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))
import pickle
from sklearn.feature_extraction.text import TfidfTransformer

#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))
# Multinomial Naive Bayes start

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#clf = MultinomialNB().fit(X_train_tfidf, training_data.flag)
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)

#SAVE MODEL
pickle.dump(clf, open("nb_model.pkl", "wb"))
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

docs_new = "Apple predicts 1 billion stock rise"
docs_new = [docs_new]

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("nb_model.pkl","rb"))

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

print("Bayesian Prediction: ",category_list[predicted[0]])




b_predicted = loaded_model.predict(X_test)
result_bayes = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': b_predicted})
result_bayes.to_csv('res_bayes.csv', sep = ',')

for predicted_item, result in zip(b_predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Confusion Matrix:\n", confusion_matrix(y_test, b_predicted))
print("Classification Report:\n", classification_report(y_test, b_predicted))
# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", accuracy_score(y_test, predicted))
# Neural Network Softmax start #############################

from sklearn.neural_network import MLPClassifier

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

clf_neural.fit(X_train, y_train)
pickle.dump(clf_neural, open("softmax.pkl", "wb"))
s_predicted = clf_neural.predict(X_test)
result_softmax = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': s_predicted})
result_softmax.to_csv('res_softmax.csv', sep = ',')

for predicted_item, result in zip(s_predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Confusion Matrix:\n", confusion_matrix(y_test, s_predicted))
print("Classification Report:\n", classification_report(y_test, s_predicted))
print("Accuracy:",metrics.accuracy_score(y_test, s_predicted))
# SVM Model start ########################################

from sklearn import svm
clf_svm = svm.LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_svm.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_svm, open("svm.pkl", "wb"))
svm_predicted = clf_svm.predict(X_test)
result_svm = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': svm_predicted})
result_svm.to_csv('res_svm.csv', sep = ',')
for predicted_item, result in zip(svm_predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predicted))
print("Classification Report:\n", classification_report(y_test,svm_predicted))
# Model Accuracy, how often is the classifier correct?
print("Accuracy: ",accuracy_score(y_test,svm_predicted))
###### Random Forest Regressor Starts ##########################

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(X_train, y_train)
reg_pred = regressor.predict(X_test)
from sklearn import metrics

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, reg_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, reg_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, reg_pred)))
###### Random Forest Classifier Starts ##########################

from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
clf_random= RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf_random.fit(X_train, y_train)

y_pred = clf.predict(X_test)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_random.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_random, open("random.pkl", "wb"))
random_pred=clf_random.predict(X_test)
result_random = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': random_pred})
result_random.to_csv('res_random.csv', sep = ',')

for predicted_item, result in zip(predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])

print("Random Forest Prediction:" + "  " + category_list[random_pred[0]])
#Import scikit-learn metrics module for confusion matrix, accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Confusion Matrix:\n", confusion_matrix(y_test, random_pred))
print("Classification Report:\n", classification_report(y_test, random_pred))
# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", accuracy_score(y_test, random_pred))

# Decision Tree Classifier starts #################

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf_decision = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

clf_decision.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_decision, open("decisionmodel.pkl", "wb"))


# train the model using the training sets y_pred=clf.predict(X_test)
decision_pred = clf_decision.predict(X_test)
result_decision = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': decision_pred})
result_decision.to_csv('res_decision.csv', sep = ',')

# for predicted_item, result in zip(predicted, y_test):
    # print(category_list[predicted_item], ' - ', category_list[result])

print( "Decision Tree Prediction:" + "  "  + category_list[decision_pred[0]])
#Import scikit-learn metrics module for confusion matrix, accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Confusion Matrix:\n", confusion_matrix(y_test, decision_pred))
print("Classification Report:\n", classification_report(y_test, decision_pred))
# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", accuracy_score(y_test, decision_pred))

### AdaBoost Classifier starts #######################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.3, random_state=42)

# Train Adaboost classifier
model = abc.fit(X_train_tfidf, training_data.flag)

pickle.dump(abc, open("adaboostmodel.pkl", "wb"))
# train the model using the training sets y_pred=clf.predict(X_test)
ada_pred = model.predict(X_test)
result_decision = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': ada_pred})
result_decision.to_csv('res_adaboost.csv', sep = ',')

# for predicted_item, result in zip(predicted, y_test):
    # print(category_list[predicted_item], ' - ', category_list[result])

print( "Ada Boost Prediction:" + "  "  + category_list[ada_pred[0]])

#Import scikit-learn metrics module for confusion matrix, accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Confusion Matrix:\n", confusion_matrix(y_test, ada_pred))
print("Classification Report:\n", classification_report(y_test, ada_pred))
# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", accuracy_score(y_test, ada_pred))
#For plotting 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score
fscore_ada = f1_score(y_test, ada_pred, average='macro')
print(fscore_ada)
plt.hist(fscore_ada, facecolor='peru',edgecolor='white', bins=10)
plt.show()
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
    
    # scores.append(score)
    # plt.hist(score, facecolor='peru',edgecolor='white', bins=10)
    
    
    fscore = f1_score(y_test, predicted, average='macro')
    precision = precision_score(y_test, predicted, average='micro')
    recall = recall_score(y_test, predicted, average='macro')
    accuracy = accuracy_score(y_test, predicted)
    conf_matrix = confusion_matrix(y_test, predicted)
    # plt.hist(fscore, label=name, facecolor=my_color,edgecolor='white', bins=10)
    
    # Print Confusion matrix, accuracy, f-score, precision and recall values per classifier
    # print("Precision:")
    # print("F-scores:"+str(fscore))
    # print(name+": "+str(accuracy))
    print(name)
    print(conf_matrix)
    # print("Recall:"+str(recall))


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
    
