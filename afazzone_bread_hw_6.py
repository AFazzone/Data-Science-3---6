# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:05:17 2020

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
Bread = pd.read_csv("BreadBasket_DMS_output.csv") 

#Create dataframe
Bread_df = df = pd.DataFrame(Bread)

#Question 1


#Drop duplicates to find unique transactions
trans = Bread_df.drop_duplicates(subset=["Transaction"])

#Find the greatest values count
hour = trans["Hour"].value_counts()
print("The busiest hour is", hour)

weekday = trans["Weekday"].value_counts()
print("The busiest weekday is ", weekday)

period = trans["Period"].value_counts()
print("The busiest period is ",period)

#Question 2

#Sum price for each group to find the greatest revenue
hour_rev = Bread_df.groupby("Hour")["Item_Price"].sum()

print("Hour revenue", hour_rev)

day_rev = Bread_df.groupby("Weekday")["Item_Price"].sum()

print("Day revenue", day_rev)

period_rev = Bread_df.groupby("Period")["Item_Price"].sum()

print("Period revenue", period_rev)




#Question 3

item = Bread_df["Item"].value_counts()
print("Popular items", item.head(5))
print("Least popular items", item.tail(10))



#Question 4
coffee_day = Bread_df[["Weekday","Transaction"]]

#Filter by day

monday = coffee_day[coffee_day["Weekday"].isin(["Monday"])]
tues = coffee_day[coffee_day["Weekday"].isin(["Tuesday"])]
weds = coffee_day[coffee_day["Weekday"].isin(["Wednesday"])]
thurs = coffee_day[coffee_day["Weekday"].isin(["Thursday"])]
friday = coffee_day[coffee_day["Weekday"].isin(["Friday"])]
saturday = coffee_day[coffee_day["Weekday"].isin(["Saturday"])]
sunday = coffee_day[coffee_day["Weekday"].isin(["Sunday"])]

#Put transactions into lists

mon_tran = monday["Transaction"].tolist()
tues_tran = tues["Transaction"].tolist()
weds_tran = weds["Transaction"].tolist()
thurs_tran = thurs["Transaction"].tolist()
frid_tran = friday["Transaction"].tolist()
sat_tran = saturday["Transaction"].tolist()
sun_tran = sunday["Transaction"].tolist()

#Put transactions into sets to remove duplicates

mon_uniq = set(mon_tran)
tues_uniq = set(tues_tran)
weds_uniq = set(weds_tran)
thurs_uniq = set(thurs_tran)
fri_uniq = set(frid_tran)
sat_uniq = set(sat_tran)
sun_uniq = set(sun_tran)

#Divide the number of transactions by number of days, then divide by 50

print("The number of baristas on Mondya is ", len(mon_uniq)/25/50)
print("The number of baristas on Tuesday is ", len(tues_uniq)/25/50)
print("The number of baristas on Wednesday is ", len(weds_uniq)/25/50)
print("The number of baristas on Thursday is ", len(thurs_uniq)/25/50)
print("The number of baristas on Friday is ", len(fri_uniq)/25/50)
print("The number of baristas on Saturday is ", len(sat_uniq)/25/50)
print("The number of baristas on Sunday is ", len(sun_uniq)/25/50)


#Question 5
#Find the average price per category
category = Bread_df.groupby("Category")["Item_Price"].mean()
print("The average price per category is", category)

#Question 6
#Find the sum of price per category
category_sum = Bread_df.groupby("Category")["Item_Price"].sum()
print("The sum of sales per category is", category_sum )


#Question 7
#Create subset
weekday_item = Bread_df[["Weekday","Item"]]

mon_item = weekday_item[weekday_item["Weekday"].isin(["Monday"])]
mon_count  = mon_item["Item"].value_counts()     

tues_item = weekday_item[weekday_item["Weekday"].isin(["Tuesday"])]
tues_count  = tues_item["Item"].value_counts()
   
weds_item = weekday_item[weekday_item["Weekday"].isin(["Wednesday"])]
weds_count  = weds_item["Item"].value_counts()
   
thurs_item = weekday_item[weekday_item["Weekday"].isin(["Thursday"])]
thurs_count  = thurs_item["Item"].value_counts()

fri_item = weekday_item[weekday_item["Weekday"].isin(["Friday"])]
fri_count  = fri_item["Item"].value_counts()

sat_item = weekday_item[weekday_item["Weekday"].isin(["Saturday"])]
sat_count  = sat_item["Item"].value_counts()

sun_item = weekday_item[weekday_item["Weekday"].isin(["Sunday"])]
sun_count  = sun_item["Item"].value_counts()
                        
print("Monday popular items", mon_count.head(5))
print("Tuesday popular items", tues_count.head(5))
print("Wednesday popular items", weds_count.head(5))
print("Thursday popular items", thurs_count.head(5))
print("Friday popular items", fri_count.head(5))
print("Saturday popular items", sat_count.head(5))
print("Sunday popular items", sun_count.head(5))


#Question 8

#Get least 5 items

print("Monday least popular items", mon_count.tail(5))
print("Tuesday least popular items", tues_count.tail(5))
print("Wednesday least popular items", weds_count.tail(5))
print("Thursday least popular items", thurs_count.tail(5))
print("Friday least popular items", fri_count.tail(5))
print("Saturday least popular items", sat_count.tail(5))
print("Sunday least popular items", sun_count.tail(5))


#Question 9

# Find the number of drinks per transaction
drink = Bread_df["Category"].value_counts()

print("Total in each category", drink)
print("The number of drinks per transaction is", 8276/9684)

