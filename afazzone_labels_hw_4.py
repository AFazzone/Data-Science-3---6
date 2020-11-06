# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:54:11 2020

@author: alf11
"""


import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from matplotlib . ticker import MaxNLocator
import seaborn as sns


# Import CSV
FDX_weeks = pd.read_csv("FDX_Weeks_labels.csv") 

# Create Dataframes
FDX_weeks_df  = pd.DataFrame(FDX_weeks)

#Filter dataframe by year

FDX_2015 = FDX_weeks_df[FDX_weeks_df["Year"].isin([2015])]
FDX_2016 = FDX_weeks_df[FDX_weeks_df["Year"].isin([2016])]


FDX_return = FDX_2016["mean_return"]



#Calculate buy and hold
 
buy_hold = []

buy_hold_value = 100
i = 0
for e in FDX_return:
    buy_hold_value = buy_hold_value + (buy_hold_value * e)
    buy_hold.append(buy_hold_value)
    i = i + 1
    
print("Buy Hold value is:", buy_hold_value)


#Calculate buying on the strategy if green value + return, if not just value

FDX_label = FDX_2016["Label"]

print(FDX_label)
print(buy_hold)
fdx_strategy_label =[]

value2 = 100
z = 49
for e in FDX_label:
   if e == "Green":
        value2 = value2 + (value2 * FDX_return[z])
        fdx_strategy_label.append(value2)
   else:
        fdx_strategy_label.append(value2)
   z = z+1
 
print("Strategy value is:", value2)


#Assign colors to values
color_dict = dict({'Green':'green',
                  'Red':'red'})

#Plot values for 2015 scatterplot

g = sns.scatterplot(x="mean_return", y="volatility", hue="Label",
             data=FDX_2015, palette = color_dict, s = 50)
plt.title('2015')
plt.show()

#Plot values for 2016 scatterplot
h = sns.scatterplot(x="mean_return", y="volatility", hue="Label",
             data=FDX_2016, palette = color_dict, s= 50)
plt.title('2016')
plt.show()

#Plot portfolio growth year 2

week = FDX_2016["Week_Number"]
weeks = np.arange(1,52)


#Calculte mean daily balance and standard deviation
avg_strat = np.mean(fdx_strategy_label)
std_strat = np.std(fdx_strategy_label)
avg_hold = np.mean(buy_hold)
std_hold = np.std(buy_hold)

print("Strat avg",avg_strat ,"std",std_strat )
print("Hold avg",avg_hold ,"std",std_hold )

x1 = weeks
y1 = buy_hold
# plotting the line 1 points 
plt.plot(x1, y1,label = "Buy & Hold: AVG = 110.43 Variance = 12.18", c = "blue")
# line 2 points
x2 = weeks
y2 = fdx_strategy_label 
# plotting the line 2 points 
plt.plot(x2, y2, c = "yellow", label = "FDX strategy: AVG = 150.41, STD = 30.09")
plt.xlabel('Weeks')
# Set the y axis label of the current axis.
plt.ylabel('Value')
# Set a title of the current axes.
plt.title('Buy & Hold vs Strategy Trading')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

#Do separate analysis on year 1 for trading labels

FDX_label_year1 = FDX_2015["Label"]
FDX_return_year1 = FDX_2015["mean_return"]

print(FDX_label)
print(buy_hold)
fdx_strategy_label_2015 =[]

value3 = 100
z = 0
for e in FDX_label_year1:
   if e == "Green":
        value3 = value3 + (value3 * FDX_return_year1[z])
        fdx_strategy_label_2015.append(value3)
   else:
        fdx_strategy_label_2015.append(value3)
   z = z+1
 
print("Strategy value is:", value3)



week_2015 = FDX_2015["Week_Number"]

x1 = week_2015
y1 = fdx_strategy_label_2015
# plotting the line 1 points 
plt.plot(x1, y1,label = "2015", c = "purple")
plt.xlabel('Weeks')
# Set the y axis label of the current axis.
plt.ylabel('Value')
# Set a title of the current axes.
plt.title('Strategy Trading Year 1')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


x2 = weeks
y2 = fdx_strategy_label 
# plotting the line 2 points 
plt.plot(x2, y2, c = "purple", label = "2016")
plt.xlabel('Weeks')
# Set the y axis label of the current axis.
plt.ylabel('Value')
# Set a title of the current axes.
plt.title('Strategy Trading Year 2')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

#Calculate how many up & down weeks in each year
print(FDX_label_year1.value_counts())
print(FDX_label.value_counts())

#Calculate the min and max of each year:
print("Min Year 1:", min(fdx_strategy_label_2015))
print("Max Year 1:", max(fdx_strategy_label_2015))

print("Min Year 2:", min(fdx_strategy_label))
print("Max Year 2:", max(fdx_strategy_label))

#Find week of Maximum

print("Year 1 index", fdx_strategy_label_2015.index(165.34776361868495))
print("Year 2 index", fdx_strategy_label.index(202.65765392420934))


#Calculate avg daily balance and vairance for year 1

avg_strat1= np.mean(fdx_strategy_label_2015)
std_strat1 = np.std(fdx_strategy_label_2015)

print("2015 AVG",avg_strat1,"Var",std_strat1 )


#final value
print("Year 1 Final", value3)
print("Year 2 Final", value2)
