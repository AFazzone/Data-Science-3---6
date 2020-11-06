# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:02:40 2020

@author: alf11
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from matplotlib . ticker import MaxNLocator
import seaborn as sns
from sklearn . metrics import mean_squared_error , r2_score

# Import CSV
FDX_data = pd.read_csv("FDX_Weeks_Labels.csv") 

#Pit into dataframe
FDX_df  = pd.DataFrame(FDX_data)

#Filter by year
FDX_2015 = FDX_df[FDX_df["Year"].isin([2015])]
FDX_2016 = FDX_df[FDX_df["Year"].isin([2016])]

# Make series week number and close price for year 1
week_num = FDX_2015["Week_Number"]
close_price = FDX_2015["Close"]
label_2015 = FDX_2015["Label"]


x = np.array(week_num)
y = np.array(close_price)


#Question 1
print("Question 1")


#Model for linear
def linear(x,y,i,label):
    degree = 1
    weights = np. polyfit (x[:i],y[:i],degree )
    model = np. poly1d (weights)
    predicted = model (x)
    rmse = np. sqrt ( mean_squared_error (y, predicted ))
    r2 = r2_score (y, predicted )
    z=1
    for e in predicted:
        if z < 49:
            if predicted[z+1] < predicted[z]:
                label.append("Red")
            else:
                label.append("Green")
            z = z +1
    return(label)


#Create lists for labels
model_1_w5 = []
model_1_w6 = []
model_1_w7 = []
model_1_w8 = []
model_1_w9 = []
model_1_w10 = []
model_1_w11 = []
model_1_w12 = []

#Run linear function for w 5-12 to get labels

linear(x, y, 6, model_1_w5)
linear(x, y, 7, model_1_w6)
linear(x, y, 8, model_1_w7)
linear(x, y, 9, model_1_w8)
linear(x, y, 10, model_1_w9)
linear(x, y, 11, model_1_w10)
linear(x, y, 12, model_1_w11)
linear(x, y, 13, model_1_w12)



#Model for quadratic
def quadratic (x,y,i,label):
    degree2 = 2
    weights2 = np. polyfit (x[:i],y[:i], degree2 )
    model2 = np. poly1d ( weights2 )
    predicted2 = model2(x)
    rmse2 = np. sqrt ( mean_squared_error (y, predicted2 ))
    r2_2 = r2_score (y, predicted2)
    z = 1
    for e in predicted2:
        if z < 49:
            if predicted2[z+1] < predicted2[z]:
                label.append("Red")
            else:
                label.append("Green")
            z=z+1
    return(label)



#Make lists for labels
model_2_w5 = []
model_2_w6 = []
model_2_w7 = []
model_2_w8 = []
model_2_w9 = []
model_2_w10 = []
model_2_w11 = []
model_2_w12 = []

#Run quadratic function fot w 5 -12 to get labels

quadratic(x, y, 6, model_2_w5)
quadratic(x, y, 7, model_2_w6)
quadratic(x, y, 8, model_2_w7)
quadratic(x, y, 9, model_2_w8)
quadratic(x, y, 10, model_2_w9)
quadratic(x, y, 11, model_2_w10)
quadratic(x, y, 12, model_2_w11)
quadratic(x, y, 13, model_2_w12)



#Model for cubic
def cubic (x,y,i,label):
    degree3 = 3
    weights3 = np. polyfit (x,y, degree3 )
    model3 = np. poly1d ( weights3 )
    predicted3 = model3(x)
    rmse3 = np. sqrt ( mean_squared_error (y, predicted3 ))
    r2_3 = r2_score (y, predicted3 )
    z=1
    for e in predicted3:
        if z < 49:
            if predicted3[z+1] < predicted3[z]:
                label.append("Red")
            else:
                label.append("Green")
            z=z+1
    return(label)



model_3_w5 = []
model_3_w6 = []
model_3_w7 = []
model_3_w8 = []
model_3_w9 = []
model_3_w10 = []
model_3_w11 = []
model_3_w12 = []

cubic(x, y, 6, model_3_w5)
cubic(x, y, 7, model_3_w6)
cubic(x, y, 8, model_3_w7)
cubic(x, y, 9, model_3_w8)
cubic(x, y, 10, model_3_w9)
cubic(x, y, 11, model_3_w10)
cubic(x, y, 12, model_3_w11)
cubic(x, y, 13, model_3_w12)

#Compute accuracy

accuracy_1 =[]
accuracy_2 =[]
accuracy_3 =[]

cp = label_2015.tolist()

def accuracy(x,y):
    i = 0
    z = 0
    for e in x:
        if x[i] == y[i]:
            z = z+1
        i=i+1
    return(z/len(x))
        

#Run accuracy function

accuracy_1.append(accuracy(cp[6:], model_1_w5))
accuracy_1.append(accuracy(cp[7:], model_1_w6))
accuracy_1.append(accuracy(cp[8:], model_1_w7))
accuracy_1.append(accuracy(cp[9:], model_1_w8))
accuracy_1.append(accuracy(cp[10:], model_1_w9))
accuracy_1.append(accuracy(cp[11:], model_1_w10))
accuracy_1.append(accuracy(cp[12:], model_1_w11))                
accuracy_1.append(accuracy(cp[13:], model_1_w12))                 
                  
accuracy_2.append(accuracy(cp[6:], model_2_w5))
accuracy_2.append(accuracy(cp[7:], model_2_w6))
accuracy_2.append(accuracy(cp[8:], model_2_w7))
accuracy_2.append(accuracy(cp[9:], model_2_w8))
accuracy_2.append(accuracy(cp[10:], model_2_w9))
accuracy_2.append(accuracy(cp[11:], model_2_w10))
accuracy_2.append(accuracy(cp[12:], model_2_w11))                  
accuracy_2.append(accuracy(cp[13:], model_2_w12))

accuracy_3.append(accuracy(cp[6:], model_3_w5))
accuracy_3.append(accuracy(cp[7:], model_3_w6))
accuracy_3.append(accuracy(cp[8:], model_3_w7))
accuracy_3.append(accuracy(cp[9:], model_3_w8))
accuracy_3.append(accuracy(cp[10:], model_3_w9))
accuracy_3.append(accuracy(cp[11:], model_3_w10))
accuracy_3.append(accuracy(cp[12:], model_3_w11))                 
accuracy_3.append(accuracy(cp[13:], model_3_w12))  

print("Accuracy 1:",accuracy_1)
print("Accuracy 2:",accuracy_2)
print("Accuracy 3:",accuracy_3)

#Plot accuracies

w = [5,6,7,8,9,10,11,12]

#Plot accuracies
x1 = w
y1 = accuracy_1
# plotting the line 1 points 
plt.plot(x1, y1, label = "Linear")

# line 2 points
x2 = w
y2 = accuracy_2
# plotting the line 2 points 
plt.plot(x2, y2, label = "Quadratic")

# line 3 points
x3 = w
y3 = accuracy_3
# plotting the line 2 points 
plt.plot(x3, y3, label = "Cubic")
plt.xlabel('Weeks')
# Set the y axis label of the current axis.
plt.ylabel('Accuracy')
# Set a title of the current axes.
plt.title('Model Accuracy')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


#Question 2

# Predict year 2

week_num2 = FDX_2016["Week_Number"]
close_price2 = FDX_2016["Close"]
label_2016 = FDX_2016["Label"]

a = np.array(week_num2)
b = np.array(close_price2)

cp2 = label_2016.tolist()


#Create lists for labels year2
model_1_w5_y2 = []
model_2_w9_y2 = []
model_3_w11_y2 = []

#Run functions for best models on year 2
linear(a, b, 6, model_1_w5_y2)
quadratic(a, b, 10, model_2_w9_y2)
cubic(a, b, 12, model_3_w11_y2)

#get accuracies for year 2 models


y2_acc_d1 = []
y2_acc_d2 = []
y2_acc_d3 = []


print("Question 2")

y2_acc_d1.append(accuracy(cp2[6:], model_1_w5_y2))
y2_acc_d2.append(accuracy(cp2[10:], model_2_w9_y2))
y2_acc_d3.append(accuracy(cp2[12:], model_3_w11_y2))

print("Year 2 accuracies:  D1 ", y2_acc_d1, "D2", y2_acc_d2, "D3", y2_acc_d3 )

#Question 3

#Function for confusion matrix

def confusion (x,y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    i=0
    for e in x: 
        if x[i] == "Green":
            if x[i] == y[i]:
                tp = tp +1
            else:
                fp = fp +1
        else:
            if x[i] == y[i]:
                tn = tn + 1
            else: 
                fn = fn +1
        i = i+1
    return("Confusion:  TP", tp, "FP", fp, "TN", tn, "FN", fn)
    
print("Question 3")
print("D1", confusion(cp2[6:], model_1_w5_y2))
print("D2", confusion(cp2[10:], model_2_w9_y2))               
print("D3", confusion(cp2[12:], model_3_w11_y2))

#Question 4


ret = FDX_2016["Return"]
ret2 = ret.tolist()

def strategy(x,a):
    value = 100
    z = 0
    for e in a:
        if e == "Green":
            value = value + ((value * x[z])/100)
        z = z + 1
    return (value)

print("Question 4")


print("D1 strategy value", strategy(ret2, model_1_w5_y2))
print("D2 strategy value", strategy(ret2, model_2_w9_y2))             
print("D3 strategy value", strategy(ret2, model_3_w11_y2))


