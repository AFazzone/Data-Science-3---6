# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 12:46:43 2020

@author: alf11
"""


import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from matplotlib . ticker import MaxNLocator
import seaborn as sns
import os
import pandas as pd
import numpy as np
import collections
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Import CSV
banknote_data = pd.read_csv("data_banknote_authentication.csv") 

#Create dataframe
banknote_df = df = pd.DataFrame(banknote_data)

#Put class into list

banknote_class = banknote_df["class"]

#Create a new list of color, 0 is green, 1 is red

banknote_color = []

for e in banknote_class:
    if e == 0:
        banknote_color.append("Green")
    else:
        banknote_color.append("Red")
        
# Add the new columns back to the dataframe

banknote_df["Color"] = banknote_color

print(banknote_df)


#Find the mean of banknote
print("Mean of all is ",np.mean(banknote_df))

#Find the standard deviation of banknote
print("Stad\ndard deviation of all is ", np.std(banknote_df))

#Filter by class

Good_df = banknote_df[(banknote_df["class"] == 0)]
Bad_df = banknote_df[(banknote_df["class"] == 1)]

#Find the mean of each class
print("Mean of Good is ",np.mean(Good_df))
print("Mean of Bad is ",np.mean(Bad_df))
      
#Find the standard deviation of banknote
print("Standard deviation of Good is ", np.std(Good_df))
print("Standard deviation of Bad is ", np.std(Bad_df))

#Question 2 Split into 2 datasets


train, test = train_test_split(banknote_data, test_size=0.5, random_state=123)

print(train)

def plot_histograms(df, cols, color):
    for n_col in cols:
        print("mean for ", n_col, " is ", df[n_col].mean())
        print("STD for ", n_col, " is ", df[n_col].std())
        df.hist(n_col, bins = 10, color=color)
    return


plot_dir = r'C:\Users\alf11'

cols = ["variance","skewness","curtosis","entropy"]


#Filter training dataset by class = 0
train_0 = train[train["class"].isin(["0"])]
plot_histograms(train_0, cols, color="green")


#Filter training dataset by class = 1
train_1 = train[train["class"].isin(["1"])]
plot_histograms(train_1, cols, color="red")

sns.pairplot(train_0)
sns.pairplot(train_1)

#plot good banknotes
colors=["green"]
sns.set_palette(sns.color_palette(colors))
print("pairwise relationships for class 0")
pair_plot_0 = sns.pairplot(train_0, vars = cols)
plt.show()
#pair_plot_0.savefig(os.path.join(plot_dir,"good_banknotes.pdf"))

#plot bad banknotes
colors=["red"]
sns.set_palette(sns.color_palette(colors))
print("pairwise relationships for class 1")
pair_plot_1 = sns.pairplot(train_1, vars = cols)
plt.show()
#pair_plot_1.savefig(os.path.join(plot_dir,"bad_banknotes.pdf"))


# Apply comparisons to test data set

test1 = np.array(test)

#Make a list of predicted values
test_predict = []
for row in test1:
    if row[0] > 2:
        test_predict.append(0)
    elif row[1] > 5:
        test_predict.append(0)
    elif row[2] > 0:
        test_predict.append(0)
    else:
        test_predict.append(1)
#Make a list of acutal values        
actual1 = test["class"].tolist()

print("Length of test",len(actual1))


#Compare predicted to actual and count tp/tn/fp/fn

count = 0
tp = 0
tn = 0
fp = 0
fn = 0



i = 0
for e in test_predict:
    if test_predict[i] == actual1[i]:
        if test_predict[i] == 0:
            tp= tp +1
        else:
            tn = tn +1
    else:
        if test_predict[i] == 1:
            fp = fp +1
        else:
            fn = fn +1
    i = i+1
        


print("TP is", tp, "FP is", fp, "TN is", tn, "FN is", fn)
        
#Question 3


train, test = train_test_split(banknote_data, test_size=0.5, random_state=123)


xtrain_k3 = train[["variance","skewness", "curtosis", "entropy"]]
ytrain_k3 = train[["class"]]

x_test = test[["variance","skewness", "curtosis", "entropy"]]
y_test = test[["class"]]

xtrain_k5 = train[["variance","skewness", "curtosis", "entropy"]]
ytrain_k5 = train[["class"]]

xtrain_k7 = train[["variance","skewness", "curtosis", "entropy"]]
ytrain_k7 = train[["class"]]

xtrain_k9 = train[["variance","skewness", "curtosis", "entropy"]]
ytrain_k9 = train[["class"]]

xtrain_k11 = train[["variance","skewness", "curtosis", "entropy"]]
ytrain_k11 = train[["class"]]

y_test_list = test["class"].tolist() 

#Run the classifier for each value of k

scaler = StandardScaler().fit(xtrain_k3)
train_k3= scaler.transform(xtrain_k3)


knn_classifier = KNeighborsClassifier(n_neighbors =3)
knn_classifier.fit(xtrain_k3,np.ravel(ytrain_k3))
pred_k3 = knn_classifier.predict (x_test)
conf_k3 = confusion_matrix(y_test_list, pred_k3, [0, 1])


knn_classifier = KNeighborsClassifier(n_neighbors =5)
knn_classifier.fit(xtrain_k5,np.ravel(ytrain_k5))
pred_k5 = knn_classifier.predict (x_test)


knn_classifier = KNeighborsClassifier(n_neighbors =7)
knn_classifier.fit(xtrain_k7,np.ravel(ytrain_k7))
pred_k7 = knn_classifier.predict (x_test)

knn_classifier = KNeighborsClassifier(n_neighbors =9)
knn_classifier.fit(xtrain_k9,np.ravel(ytrain_k9))
pred_k9 = knn_classifier.predict (x_test)

knn_classifier = KNeighborsClassifier(n_neighbors =11)
knn_classifier.fit(xtrain_k11,np.ravel(ytrain_k11))
pred_k11 = knn_classifier.predict (x_test)

#Compute accuray

def accuracy (predict, test):
    i = 0
    kcount_correct = 0
    for e in predict:
        if predict[i] == test[i]:
            kcount_correct = kcount_correct + 1
        i = i +1
    return kcount_correct

print("K3 accuracy",(accuracy(pred_k3, y_test_list)/686))
print("K5 accuracy",(accuracy(pred_k5, y_test_list)/686))
print("K7 accuracy",(accuracy(pred_k7, y_test_list)/686))
print("K9 accuracy",(accuracy(pred_k9, y_test_list)/686))
print("K11 accuracy",(accuracy(pred_k11, y_test_list)/686))


print(conf_k3)

#Plot accuracy
x1 = [3,5,7,9,11]
y1 = [1, .992, .986, .986, .981]
# plotting the line 1 points 
plt.plot(x1, y1, label = 'KNN Accuracy')
plt.xlabel('KNN')
# Set the y axis label of the current axis.
plt.ylabel('Accuracy')
# Set a title of the current axes.
plt.title('KNN Accuracy')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

#Predict value of my BUID

new_instance = np. asmatrix ([4, 3, 6 , 1])
new_instance_scaled = scaler . transform ( new_instance )
prediction = knn_classifier . predict ( new_instance_scaled )

print("Prediction is", prediction)

#Question 4

A, B= train_test_split(banknote_data, test_size=0.5, random_state=123)

#Make subsets
a1 = A[["skewness", "curtosis", "entropy"]]
a2 = A[["variance", "curtosis", "entropy"]]
a3 = A[["variance","skewness", "entropy"]]
a4 = A[["variance","skewness", "curtosis"]]

b1 = B[["skewness", "curtosis", "entropy"]]
b2 = B[["variance", "curtosis", "entropy"]]
b3 = B[["variance","skewness", "entropy"]]
b4 = B[["variance","skewness", "curtosis"]]

y_train = A[["class"]]

knn_classifier = KNeighborsClassifier(n_neighbors =3)
knn_classifier.fit(a1,np.ravel(y_train))
pred_a1 = knn_classifier.predict (b1)
conf_a1 = confusion_matrix(y_test_list, pred_k3, [0, 1])


knn_classifier = KNeighborsClassifier(n_neighbors =3)
knn_classifier.fit(a2,np.ravel(y_train))
pred_a2 = knn_classifier.predict (b2)
conf_a2 = confusion_matrix(y_test_list, pred_a2, [0, 1])

knn_classifier = KNeighborsClassifier(n_neighbors =3)
knn_classifier.fit(a3,np.ravel(y_train))
pred_a3 = knn_classifier.predict (b3)
conf_a3 = confusion_matrix(y_test_list, pred_a3, [0, 1])


knn_classifier = KNeighborsClassifier(n_neighbors =3)
knn_classifier.fit(a4,np.ravel(y_train))
pred_a4 = knn_classifier.predict (b4)
conf_a4 = confusion_matrix(y_test_list, pred_a4, [0, 1])

print("A1 Confusion", conf_a1)
print("A2 Confusion", conf_a2)
print("A3 Confusion", conf_a3)
print("A4 Confusion", conf_a4)


#Question5
#Split train test
X, Y = train_test_split(banknote_data, test_size=0.5, random_state=123)

xtrain_log = X[["variance","skewness", "curtosis", "entropy"]]
ytrain_log = X[["class"]]

xtest_log = Y[["variance","skewness", "curtosis", "entropy"]]
ytest_log = Y[["class"]]

#Run Classifier
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(xtrain_log,ytrain_log)
prediction = log_reg_classifier . predict (xtest_log )
accuracy = log_reg_classifier.score (xtrain_log, np.ravel(ytrain_log))
conf_log = confusion_matrix(y_test_list, prediction, [0, 1])

print("Log Confusion is", conf_log)

#Predict new instance

new_x = scaler.transform(np.asmatrix ([4, 3, 6 , 1]))
log_predicted = log_reg_classifier.predict ( new_x )

print("log new instance predicted,", log_predicted)

#Question 6

#Make subsets
f1 = X[["skewness", "curtosis", "entropy"]]
f2 = X[["variance", "curtosis", "entropy"]]
f3 = X[["variance","skewness", "entropy"]]
f4 = X[["variance","skewness", "curtosis"]]

t1 = Y[["skewness", "curtosis", "entropy"]]
t2 = Y[["variance", "curtosis", "entropy"]]
t3 = Y[["variance","skewness", "entropy"]]
t4 = Y[["variance","skewness", "curtosis"]]

#Run Classifier on subsets
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(f1,ytrain_log)
prediction = log_reg_classifier . predict (t1)
f_1 = confusion_matrix(y_test_list, prediction, [0, 1])

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(f2,ytrain_log)
prediction = log_reg_classifier . predict (t2)
f_2 = confusion_matrix(y_test_list, prediction, [0, 1])


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(f3,ytrain_log)
prediction = log_reg_classifier . predict (t3)
f_3 = confusion_matrix(y_test_list, prediction, [0, 1])


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(f4,ytrain_log)
prediction = log_reg_classifier . predict (t4)
f_4 = confusion_matrix(y_test_list, prediction, [0, 1])

print("F1 Confusion", f_1)
print("F2 Confusion", f_2)
print("F3 Confusion", f_3)
print("F4 Confusion", f_4)





