# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 05:32:22 2020

@author: alf11
"""

import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from matplotlib . ticker import MaxNLocator
import seaborn as sns


# Import CSV
FDX_data = pd.read_csv("FDX.csv") 
SPY_data = pd.read_csv("SPY.csv") 

# Create Dataframes
FDX_df  = pd.DataFrame(FDX_data)
SPY_df  = pd.DataFrame(SPY_data)


#Filter FDX dataframe by year
FDX_2015 = FDX_df[FDX_df["Year"].isin([2015])]
FDX_2016 = FDX_df[FDX_df["Year"].isin([2016])]
FDX_2017 = FDX_df[FDX_df["Year"].isin([2017])]
FDX_2018 = FDX_df[FDX_df["Year"].isin([2018])]
FDX_2019 = FDX_df[FDX_df["Year"].isin([2019])]

#calculate Mean for each year
Mean_FDX_2015 = FDX_2015.groupby("Weekday")["Return"].mean()
print("FDX 2015 mean",Mean_FDX_2015)

Mean_FDX_2016 = FDX_2016.groupby("Weekday")["Return"].mean()
print("FDX 2016 mean",Mean_FDX_2016)

Mean_FDX_2017 = FDX_2017.groupby("Weekday")["Return"].mean()
print("FDX 2017 mean",Mean_FDX_2017)

Mean_FDX_2018 = FDX_2018.groupby("Weekday")["Return"].mean()
print("FDX 2018 mean",Mean_FDX_2018)

Mean_FDX_2019 = FDX_2019.groupby("Weekday")["Return"].mean()
print("FDX 2019 mean",Mean_FDX_2019)

#calculate standard deviation for each year
STD_FDX_2015 = FDX_2015.groupby("Weekday")["Return"].std()
print("FDX 2015 standard deviation",STD_FDX_2015)

STD_FDX_2016 = FDX_2016.groupby("Weekday")["Return"].std()
print("FDX 2016 standard deviation",STD_FDX_2016)

STD_FDX_2017 = FDX_2017.groupby("Weekday")["Return"].std()
print("FDX 2017 standard deviation",STD_FDX_2017)

STD_FDX_2018 = FDX_2018.groupby("Weekday")["Return"].std()
print("FDX 2018 standard deviation",STD_FDX_2018)

STD_FDX_2019 = FDX_2019.groupby("Weekday")["Return"].std()
print("FDX 2019 standard deviation",STD_FDX_2019)


# Create positive subsets, count how many each day

FDX_2015_R=FDX_2015[FDX_2015["Return"]>=0]
R_count_2015 = FDX_2015_R.groupby("Weekday")["Return"].count()
print("Positive return 2015",R_count_2015)

FDX_2016_R=FDX_2016[FDX_2016["Return"]>=0]
R_count_2016 = FDX_2016_R.groupby("Weekday")["Return"].count()
print("Positive return 2016",R_count_2016)

FDX_2017_R=FDX_2017[FDX_2017["Return"]>=0]
R_count_2015 = FDX_2015_R.groupby("Weekday")["Return"].count()
print("Positive return 2017",R_count_2015)


FDX_2018_R=FDX_2018[FDX_2018["Return"]>=0]
R_count_2018 = FDX_2018_R.groupby("Weekday")["Return"].count()
print("Positive return 2018",R_count_2018)

FDX_2019_R=FDX_2019[FDX_2019["Return"]>=0]
R_count_2019 = FDX_2019_R.groupby("Weekday")["Return"].count()
print("Positive return 2019",R_count_2019)

# Create negative subsets, count how many each day


FDX_2015_R_neg=FDX_2015[FDX_2015["Return"]<0]
R_count_2015 = FDX_2015_R_neg.groupby("Weekday")["Return"].count()
print("Negative return 2015",R_count_2015)

FDX_2016_R_neg=FDX_2016[FDX_2016["Return"]<0]
R_count_2016 = FDX_2016_R_neg.groupby("Weekday")["Return"].count()
print("Negative return 2016",R_count_2016)

FDX_2017_R_neg=FDX_2017[FDX_2017["Return"]<0]
R_count_2017 = FDX_2017_R_neg.groupby("Weekday")["Return"].count()
print("Negative return 2017",R_count_2017)

FDX_2018_R_neg=FDX_2018[FDX_2018["Return"]<0]
R_count_2018 = FDX_2018_R_neg.groupby("Weekday")["Return"].count()
print("Negative return 2018",R_count_2018)

FDX_2019_R_neg=FDX_2019[FDX_2019["Return"]<0]
R_count_2019 = FDX_2019_R_neg.groupby("Weekday")["Return"].count()
print("Negative return 2019",R_count_2019)

#calculate Mean for each positive subset

Mean_FDX_2015_R = FDX_2015_R.groupby("Weekday")["Return"].mean()
print("FDX 2015 positive mean",Mean_FDX_2015_R)

Mean_FDX_2016_R = FDX_2016_R.groupby("Weekday")["Return"].mean()
print("FDX 2016 positive mean",Mean_FDX_2016_R)

Mean_FDX_2017_R = FDX_2017_R.groupby("Weekday")["Return"].mean()
print("FDX 2017 positive mean",Mean_FDX_2017_R)

Mean_FDX_2018_R = FDX_2018_R.groupby("Weekday")["Return"].mean()
print("FDX 2018 positive mean",Mean_FDX_2018_R)

Mean_FDX_2019_R = FDX_2019_R.groupby("Weekday")["Return"].mean()
print("FDX 2019 positive mean",Mean_FDX_2019_R)

#calculate Mean for each negative subset


Mean_FDX_2015_R_neg = FDX_2015_R_neg.groupby("Weekday")["Return"].mean()
print("FDX 2015 negative mean",Mean_FDX_2015_R_neg)

Mean_FDX_2016_R_neg = FDX_2016_R_neg.groupby("Weekday")["Return"].mean()
print("FDX 2016 negative mean",Mean_FDX_2016_R_neg)

Mean_FDX_2017_R_neg = FDX_2017_R_neg.groupby("Weekday")["Return"].mean()
print("FDX 2017 negative mean",Mean_FDX_2017_R_neg)

Mean_FDX_2018_R_neg = FDX_2018_R_neg.groupby("Weekday")["Return"].mean()
print("FDX 2018 negative mean",Mean_FDX_2018_R_neg)

Mean_FDX_2019_R_neg = FDX_2019_R_neg.groupby("Weekday")["Return"].mean()
print("FDX 2019 negative mean",Mean_FDX_2019_R_neg)



#calculate STD for each positive subset

STD_FDX_2015_R = FDX_2015_R.groupby("Weekday")["Return"].std()
print("FDX 2015 positive std",STD_FDX_2015_R)

STD_FDX_2016_R = FDX_2016_R.groupby("Weekday")["Return"].std()
print("FDX 2016 positive std",STD_FDX_2016_R)

STD_FDX_2017_R = FDX_2017_R.groupby("Weekday")["Return"].std()
print("FDX 2017 positive std",STD_FDX_2017_R)

STD_FDX_2018_R = FDX_2018_R.groupby("Weekday")["Return"].std()
print("FDX 2018 positive std",STD_FDX_2018_R)

STD_FDX_2019_R = FDX_2019_R.groupby("Weekday")["Return"].std()
print("FDX 2019 positive std",STD_FDX_2019_R)


#calculate STD for each negative subset


STD_FDX_2015_R_neg = FDX_2015_R_neg.groupby("Weekday")["Return"].std()
print("FDX 2015 negative std",STD_FDX_2015_R_neg)

STD_FDX_2016_R_neg = FDX_2016_R_neg.groupby("Weekday")["Return"].std()
print("FDX 2016 negative std",STD_FDX_2016_R_neg)

STD_FDX_2017_R_neg = FDX_2017_R_neg.groupby("Weekday")["Return"].std()
print("FDX 2017 negative std",STD_FDX_2017_R_neg)

STD_FDX_2018_R_neg = FDX_2018_R_neg.groupby("Weekday")["Return"].std()
print("FDX 2018 negative std",STD_FDX_2018_R_neg)

STD_FDX_2019_R_neg = FDX_2019_R_neg.groupby("Weekday")["Return"].std()
print("FDX 2019 negative std",STD_FDX_2019_R_neg)
    




print("Question 3: ")

# Calculate mean for all FDX
Mean_FDX = FDX_df.groupby("Weekday")["Return"].mean()
print("FDX mean",Mean_FDX)


#calculate standard deviation for all FDX
STD_FDX = FDX_df.groupby("Weekday")["Return"].std()
print("FDX standard deviation",STD_FDX)

#Calculate negative subset for all FDX
FDX_R_neg=FDX_df[FDX_df["Return"]<0]
FDX_R_count = FDX_R_neg.groupby("Weekday")["Return"].count()
print("Negative return FDX ",FDX_R_count)

# Create positive subsets for all FDX

FDX_R = FDX_df[FDX_df["Return"]>=0]
FDX_R_count = FDX_R.groupby("Weekday")["Return"].count()
print("Positive return",FDX_R_count)

#calculate Mean for each positive subset

Mean_FDX_R = FDX_R.groupby("Weekday")["Return"].mean()
print("FDX positive mean",Mean_FDX_R)

#calculate Mean for each negative subset

Mean_FDX_R_neg = FDX_R_neg.groupby("Weekday")["Return"].mean()
print("FDX negative mean",Mean_FDX_R_neg)

#calculate STD for each positive subset

STD_FDX_R = FDX_R.groupby("Weekday")["Return"].std()
print("FDX positive std",STD_FDX_R)

#calculate STD for each negative subset

STD_FDX_R_neg = FDX_R_neg.groupby("Weekday")["Return"].std()
print("FDX negative std",STD_FDX_R_neg)


# Calculate mean for all SPY
Mean_SPY = SPY_df.groupby("Weekday")["Return"].mean()
print("SPY mean",Mean_SPY)


#calculate standard deviation for all SPY
STD_SPY = SPY_df.groupby("Weekday")["Return"].std()
print("SPY standard deviation",STD_SPY)

#Calculate negative subset for all SPY
SPY_R_neg=SPY_df[SPY_df["Return"]<0]
SPY_R_count = SPY_R_neg.groupby("Weekday")["Return"].count()
print("Negative return SPY ",SPY_R_count)

# Create positive subsets for all SPY

SPY_R = SPY_df[SPY_df["Return"]>=0]
SPY_R_count = SPY_R.groupby("Weekday")["Return"].count()
print("Positive return",SPY_R_count)

#calculate Mean for each positive subset

Mean_SPY_R = SPY_R.groupby("Weekday")["Return"].mean()
print("SPY positive mean",Mean_SPY_R)

#calculate Mean for each negative subset

Mean_SPY_R_neg = SPY_R_neg.groupby("Weekday")["Return"].mean()
print("SPY negative mean",Mean_SPY_R_neg)

#calculate STD for each positive subset

STD_SPY_R = SPY_R.groupby("Weekday")["Return"].std()
print("SPY positive std",STD_SPY_R)

#calculate STD for each negative subset

STD_SPY_R_neg = SPY_R_neg.groupby("Weekday")["Return"].std()
print("SPY negative std",STD_SPY_R_neg)

#Calculate Oracle

FDX_return = FDX_df["Return"]
SPY_return = SPY_df["Return"]

FDX_oracle_value = 100
SPY_oracle_value = 100

for e in FDX_return:
    if e > 0:
        FDX_oracle_value = FDX_oracle_value + (e * FDX_oracle_value)

for e in SPY_return:
    if e > 0:
        SPY_oracle_value = SPY_oracle_value + (e * SPY_oracle_value)

print("FDX oracle is:", FDX_oracle_value)
print("SPY oracle is:", SPY_oracle_value)

#Calculate buy and hold

buy_hold_value = 100

for e in FDX_return:
    buy_hold_value = buy_hold_value + (e * buy_hold_value)
    
print("Buy Hold is:", buy_hold_value)


# Find top 10 returns

FDX_list =[]

for e in FDX_return:
    FDX_list.append(e)
    

sort_FDX = FDX_list.copy()



sort_FDX.sort()


print ("Bottom 10: ",sort_FDX[:10])
print ("Top 10: ",sort_FDX[-10:])

#Find indexes of top 10 and bottom 10

fdx_top = sort_FDX[-10:]
fdx_bottom = sort_FDX[:10]

fdx_top_index= []


for e in fdx_top:
    fdx_top_index.append(FDX_list.index(e))
    
print("Top index", fdx_top_index)

fdx_bottom_index = []

for e in fdx_bottom:
    fdx_bottom_index.append(FDX_list.index(e))

print("Bottom indexes", fdx_bottom_index)    

#Make a list without top ten

FDX_list_top_10 = []

for e in FDX_list:
    if FDX_list.index(e) not in fdx_top_index:
        FDX_list_top_10.append(e)
        
#Calculate  total without top ten        


fdx_top_value = 100

for e in FDX_list_top_10:
    if e > 0:
        fdx_top_value = fdx_top_value + (e * fdx_top_value)
        
print ("FDX Top Value", fdx_top_value)

#Make a list without bottom ten

FDX_list_bottom_10 = []


for e in FDX_list:
    if FDX_list.index(e) not in fdx_bottom_index:
        FDX_list_bottom_10.append(e)


fdx_bottom_value = 100

for e in FDX_list_bottom_10:
    if e > 0:
        fdx_bottom_value = fdx_bottom_value + (e * fdx_bottom_value)
        
print ("FDX Bottom Value", fdx_bottom_value)

#Make a list without  top 5 and bottom 5

FDX_half = fdx_top_index[-5:] + fdx_bottom_index[:5]


FDX_list_half = []


for e in FDX_list:
    if FDX_list.index(e) not in FDX_half:
        FDX_list_half.append(e)

fdx_half_value = 100

for e in FDX_list_half:
    if e > 0:
        fdx_half_value = fdx_half_value + (e * fdx_half_value)

print("FDX wihtout the top and bottom 5", fdx_half_value)


# Find top 10 returns

SPY_list =[]

for e in SPY_return:
    SPY_list.append(e)
    

sort_SPY = SPY_list.copy()



sort_SPY.sort()


print ("Bottom 10: ",sort_SPY[:10])
print ("Top 10: ",sort_SPY[-10:])

#Find indexes of top 10 and bottom 10

spy_top = sort_SPY[-10:]
spy_bottom = sort_SPY[:10]

spy_top_index= []


for e in spy_top:
    spy_top_index.append(SPY_list.index(e))
    
print("Top index", spy_top_index)

spy_bottom_index = []

for e in spy_bottom:
    spy_bottom_index.append(SPY_list.index(e))

print("Bottom indexes", spy_bottom_index)    

#Make a list without top ten

SPY_list_top_10 = []

for e in SPY_list:
    if SPY_list.index(e) not in spy_top_index:
        SPY_list_top_10.append(e)
        
#Calculate  total without top ten        


spy_top_value = 100

for e in SPY_list_top_10:
    if e > 0:
        spy_top_value = spy_top_value + (e * spy_top_value)
        
print ("SPY Top Value", spy_top_value)

#Make a list without bottom ten

SPY_list_bottom_10 = []


for e in SPY_list:
    if SPY_list.index(e) not in spy_bottom_index:
        SPY_list_bottom_10.append(e)


spy_bottom_value = 100

for e in SPY_list_bottom_10:
    if e > 0:
        spy_bottom_value = spy_bottom_value + (e * spy_bottom_value)
        
print ("SPY Bottom Value", spy_bottom_value)

#Make a list without  top 5 and bottom 5

SPY_half = spy_top_index[-5:] + spy_bottom_index[:5]


SPY_list_half = []


for e in SPY_list:
    if SPY_list.index(e) not in SPY_half:
        SPY_list_half.append(e)

spy_half_value = 100

for e in SPY_list_half:
    if e > 0:
        spy_half_value = spy_half_value + (e * spy_half_value)

print("SPY wihtout the top and bottom 5", spy_half_value)