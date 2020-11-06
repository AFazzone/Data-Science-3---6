# -*- coding: utf-8 -# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 12:46:43 2020

@author: alf11
"""

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from matplotlib . ticker import MaxNLocator
import seaborn as sns
import os
import collections
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.metrics import confusion_matrix




# Import CSV
FDX_weeks = pd.read_csv("FDX_Weeks_labels.csv") 

# Create Dataframes
FDX_weeks_df  = pd.DataFrame(FDX_weeks)

#Filter dataframe by year

FDX_2015 = FDX_weeks_df[FDX_weeks_df["Year"].isin([2015])]
FDX_2016 = FDX_weeks_df[FDX_weeks_df["Year"].isin([2016])]


#KNN Assignment

train, test = train_test_split(FDX_2016, test_size=0.5, random_state=123)

xtrain_k3 = train[["mean_return", "volatility"]]
ytrain_k3 = test[["Label"]]

x_test = test[["mean_return", "volatility"]]
y_test = test[["Label"]]

xtrain_k5 = train[["mean_return", "volatility"]]
ytrain_k5 = train[["Label"]]

xtrain_k7 = train[["mean_return", "volatility"]]
ytrain_k7 = train[["Label"]]

xtrain_k9 = train[["mean_return", "volatility"]]
ytrain_k9 = train[["Label"]]

xtrain_k11 = train[["mean_return", "volatility"]]
ytrain_k11 = train[["Label"]]

y_test_list = test["Label"].tolist() 


#Run the classifier for each value of k

scaler = StandardScaler().fit(xtrain_k3)
train_k3= scaler.transform(xtrain_k3)


knn_classifier = KNeighborsClassifier(n_neighbors =3)
knn_classifier.fit(xtrain_k3,np.ravel(ytrain_k3))
pred_k3 = knn_classifier.predict (x_test)
conf_k3 = confusion_matrix(y_test_list, pred_k3, ["Green", "Red"])
k3_accuracy = knn_classifier.score (xtrain_k3, np.ravel(ytrain_k3))

print("K3", conf_k3)


knn_classifier = KNeighborsClassifier(n_neighbors =5)
knn_classifier.fit(xtrain_k5,np.ravel(ytrain_k5))
pred_k5 = knn_classifier.predict (x_test)
conf_k5 = confusion_matrix(y_test_list, pred_k5, ["Green", "Red"])
k5_accuracy = knn_classifier.score (xtrain_k5, np.ravel(ytrain_k5))

print("K5", conf_k5)


knn_classifier = KNeighborsClassifier(n_neighbors =7)
knn_classifier.fit(xtrain_k7,np.ravel(ytrain_k7))
pred_k7 = knn_classifier.predict (x_test)
conf_k7 = confusion_matrix(y_test_list, pred_k7, ["Green", "Red"])
k7_accuracy = knn_classifier.score (xtrain_k7, np.ravel(ytrain_k7))

print("K7", conf_k7)


knn_classifier = KNeighborsClassifier(n_neighbors =9)
knn_classifier.fit(xtrain_k9,np.ravel(ytrain_k9))
pred_k9 = knn_classifier.predict (x_test)
conf_k9 = confusion_matrix(y_test_list, pred_k9, ["Green", "Red"])
k9_accuracy = knn_classifier.score (xtrain_k9, np.ravel(ytrain_k9))

print("K9", conf_k9)


knn_classifier = KNeighborsClassifier(n_neighbors =11)
knn_classifier.fit(xtrain_k11,np.ravel(ytrain_k11))
pred_k11 = knn_classifier.predict (x_test)
conf_k11 = confusion_matrix(y_test_list, pred_k11, ["Green", "Red"])
k11_accuracy = knn_classifier.score (xtrain_k11, np.ravel(ytrain_k11))

print("K11", conf_k11)


#Plot accuracy
x1 = [3,5,7,9,11]
y1 = [.72, .92, .92, .88, .88]
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

#Rub year 1 versus year 2 for k5

K5_x_train = FDX_2015[["mean_return", "volatility"]]
K5_y_train = FDX_2015[["Label"]] 


K5_x_test = FDX_2016[["mean_return", "volatility"]]
K5_y_test = FDX_2016[["Label"]] 

test_list = FDX_2016["Label"].tolist() 


knn_classifier = KNeighborsClassifier(n_neighbors =5)
knn_classifier.fit(K5_x_train,np.ravel(K5_y_train))
pred_k5 = knn_classifier.predict (K5_x_test)
conf_k5 = confusion_matrix(test_list, pred_k5, ["Green", "Red"])


print("K5 year 2", conf_k5)



#Calculate buy and hold
FDX_return = FDX_2016["mean_return"]
 

buy_hold = []

buy_hold_value = 100
i = 0
for e in FDX_return:
    buy_hold_value = buy_hold_value + (buy_hold_value * e)
    buy_hold.append(buy_hold_value)
    i = i + 1
    
print("Buy Hold value is:", buy_hold_value)


#Calculate buying on the strategy if green value + return, if not just value

FDX_label = pred_k5

FDX_return = FDX_return.tolist()

value2 = 100

z = 0
for e in FDX_label:
   if e == "Green":
        value2 = value2 + (value2 * FDX_return[z])
   z = z + 1
 

print("Strategy value is:", value2)


#Logistic Problem

#Split train test
X, Y = train_test_split(FDX_2015, test_size=0.5, random_state=123)

xtrain_log = X[["mean_return", "volatility"]]
ytrain_log = X[["Label"]]

xtest_log = Y[["mean_return", "volatility"]]
ytest_log = Y[["Label"]]

#Run Classifier on year 1
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(xtrain_log,ytrain_log)
prediction = log_reg_classifier . predict (xtest_log )
accuracy = log_reg_classifier.score (xtrain_log, np.ravel(ytrain_log))
conf_log = confusion_matrix(y_test_list, prediction, ["Green", "Red"])

import statsmodels.api as sm
logit_model=sm.Logit(y1,x1)
result=logit_model.fit()
print(result.summary())



#Run Classifier on year 2

x2015_log = FDX_2015[["mean_return", "volatility"]]
y2015_log = FDX_2015[["Label"]]

x2016_log = FDX_2016[["mean_return", "volatility"]]
y2016_log = FDX_2016[["Label"]]



log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(x2015_log,y2015_log)
prediction = log_reg_classifier . predict (x2016_log )
accuracy = log_reg_classifier.score (x2015_log, np.ravel(y2015_log))
conf_log = confusion_matrix(test_list, prediction, ["Green", "Red"])

print("Year 2 log confusion", conf_log)


#Calculate buying on the strategy if green value + return, if not just value



value = 100

z = 0
for e in prediction:
   if e == "Green":
        value = value + (value * FDX_return[z])
   z = z + 1
 

print("Strategy value is:", value2)



#Separate points 

# Import CSV
FDX_weeks2 = pd.read_csv("Reduced_points_FDX_Weeks_labels.csv") 

# Create Dataframes
Reduced = pd.DataFrame(FDX_weeks2)


New = Reduced[Reduced["Year"].isin([2015])]

#Plot values for new 2015 scatterplot

#Assign colors to values
color_dict = dict({'Green':'green',
                  'Red':'red'})


g = sns.scatterplot(x="mean_return", y="volatility", hue="Label",
             data=New, palette = color_dict, s = 50)
plt.title('2015')
plt.show()


New_2016 = New = Reduced[Reduced["Year"].isin([2016])]

a = New_2016["mean_return"].tolist()

b = New_2016["volatility"].tolist()

label_list =[]

#calculate aabove or below the line, assign label

i = 0
for e in a:
    if (a[i]*.5 +.01) > b[i]:
        label_list.append("red")
    else:
        label_list.append("green")   
    i = i + 1


#Calculate Strategy
value3 = 100
z = 0
for e in label_list:
   if e == "green":
        value3 = value3 + (value3 * a[z])
   z = z + 1

print("Separate value stratgy", value3)








