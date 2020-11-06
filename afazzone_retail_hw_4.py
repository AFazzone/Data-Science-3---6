# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:17:30 2020

@author: alf11
"""
import math
import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
from matplotlib . ticker import MaxNLocator
import seaborn as sns


# Import CSV
retail_data = pd.read_csv("online_retail.csv") 

# Enter csv into dataframe
retail_df  = pd.DataFrame(retail_data)

#Filter retail dataframe by price:
unit_price = retail_df['UnitPrice'].values.tolist()

# Create separate lists for each dollar price
price_1 = []
price_2 = []
price_3 = []
price_4 = []
price_5 = []
price_6 = []
price_7 = []
price_8 = []
price_9 = []

# Append each price to the appropriate list

for e in unit_price:
    if round(e, 0) == 1:
        price_1.append(e)
    if round(e, 0) == 2:
        price_2.append(e)
    if round(e, 0) == 3:
        price_3.append(e)
    if round(e, 0) == 4:
        price_4.append(e)
    if round(e, 0) == 5:
        price_5.append(e)
    if round(e, 0) == 6:
        price_6.append(e)
    if round(e, 0) == 7:
        price_7.append(e)
    if round(e, 0) == 8:
        price_8.append(e)
    if round(e, 0) == 9:
        price_9.append(e)
        
        
print("The length of price1 is", len(price_1))
print("The length of price2 is", len(price_2))
print("The length of price3 is", len(price_3))
print("The length of price4 is", len(price_4))
print("The length of price5 is", len(price_5))
print("The length of price6 is", len(price_6))
print("The length of price7 is", len(price_7))
print("The length of price8 is", len(price_8))
print("The length of price9 is", len(price_9))

#create list of totals

totals_list = [len(price_1),len(price_2), len(price_3), len(price_4),
               len(price_5), len(price_6), len(price_7), len(price_8),
               len(price_9)]


#sum list of totals

total_digits = 0

for e  in totals_list:
    total_digits = total_digits + e


dist_list = [(len(price_1)/total_digits),(len(price_2)/total_digits),
             (len(price_3)/total_digits), (len(price_4)/total_digits),
             (len(price_5)/total_digits), (len(price_6)/total_digits), 
             (len(price_7)/total_digits), (len(price_8)/total_digits),
             (len(price_9)/total_digits)]

print("The retail distribution is :", dist_list)

# rounded distribution percentage

# Enter other distributions into list

dist_list_2 = [28.4, 25.6, 16.2, 10.9, 5.8, 4.6, 1.9, 6.2, .4]

bernford = [30.1, 17.6, 12.5, 9.7, 7.9, 6.5, 5.8, 5.1, 4.6]


equal_weight = [11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1]
x_axis = [1,2,3,4,5,6,7,8,9]

#Plot Distributions

plt.bar(x_axis, height = [30.1, 17.6, 12.5, 9.7, 7.9, 6.5, 5.8, 5.1, 4.6] )
plt.xticks(x_axis)
plt.ylabel('Bernford')
plt.title('Model 2 -  Bernford Real Distribution')
plt.show()

plt.bar(x_axis, height = [11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1] )
plt.xticks(x_axis)
plt.ylabel('Equal weights')
plt.title('Model 1 - Equal Weight - Real Distribution')
plt.show()


plt.bar(x_axis, height = [28.4, 25.6, 16.2, 10.9, 5.8, 4.6, 1.9, 6.2, .4] )
plt.xticks(x_axis)
plt.ylabel('Retail Price')
plt.title('Retail price vaules 1 - 9 - Real Distribution')
plt.show()


# Create a dataframe of the distributions

df_dist = pd.DataFrame(dist_list_2, index =[1,2,3,4,5,6,7,8,9], 
                                              columns =['Retail']) 
df_dist ['Bernford'] = bernford
df_dist ['Equal Weight'] = equal_weight

# Calculate error columns
 
df_dist['Equal Error'] = abs(df_dist['Retail'] - df_dist['Equal Weight'])/df_dist['Retail']

df_dist['Bernford Error'] = abs(df_dist['Retail'] - df_dist['Bernford'])/df_dist['Retail']
  
print(df_dist)

#Plot Errors for each model

plt.bar(x_axis, height = df_dist['Equal Error'] )
plt.xticks(x_axis)
plt.ylabel('Equal Weight Errors')
plt.title('Model 1 - Equal Weight - Errors')
plt.show()


plt.bar(x_axis, height = df_dist['Bernford Error']   )
plt.xticks(x_axis)
plt.ylabel('Burnford Errors')
plt.title('Model 2 - Burnford - Errors')
plt.show()


# Calculate root mean squared

df_dist['Equal Error Squared'] = (df_dist['Retail'] - df_dist['Equal Weight'])**2
df_dist['Bernford Error Squared'] = (df_dist['Retail'] - df_dist['Bernford'])**2

bernford_sq = df_dist['Bernford Error Squared']
equal_sq = df_dist['Equal Error Squared']

sum_bernford_sq = 0
sum_equal_sq = 0

for e in bernford_sq:
    sum_bernford_sq = sum_bernford_sq + e
    
Rt_mn_sq_bernford = math.sqrt(sum_bernford_sq/9)
    
print("RMSE bernford is", Rt_mn_sq_bernford)

for e in equal_sq:
    sum_equal_sq = sum_equal_sq + e
    
Rt_mn_sq_equal = math.sqrt(sum_equal_sq/9)

print("RMSE equal is", Rt_mn_sq_equal)




#Filter retail dataframe by country
Italy_df = retail_df[retail_df["Country"].isin(["Italy"])]
Israel_df = retail_df[retail_df["Country"].isin(["Israel"])]
Singapore_df = retail_df[retail_df["Country"].isin(["Singapore"])]



#Filter retail dataframe by price_Italy_:
unit_price_Italy_ = Italy_df['UnitPrice'].values.tolist()

# Create separate lists for each dollar price_Italy_
price_Italy__1 = []
price_Italy__2 = []
price_Italy__3 = []
price_Italy__4 = []
price_Italy__5 = []
price_Italy__6 = []
price_Italy__7 = []
price_Italy__8 = []
price_Italy__9 = []



#Filter retail dataframe by price_Singapore_:
unit_price_Singapore_ = Singapore_df['UnitPrice'].values.tolist()

# Create separate lists for each dollar price_Singapore_
price_Singapore__1 = []
price_Singapore__2 = []
price_Singapore__3 = []
price_Singapore__4 = []
price_Singapore__5 = []
price_Singapore__6 = []
price_Singapore__7 = []
price_Singapore__8 = []
price_Singapore__9 = []


#Filter retail dataframe by price_Israel_:
unit_price_Israel_ = Israel_df['UnitPrice'].values.tolist()

# Create separate lists for each dollar price_Israel_
price_Israel__1 = []
price_Israel__2 = []
price_Israel__3 = []
price_Israel__4 = []
price_Israel__5 = []
price_Israel__6 = []
price_Israel__7 = []
price_Israel__8 = []
price_Israel__9 = []

# Append to appropriate List

for e in unit_price_Israel_:
    if round(e, 0) == 1:
        price_Israel__1.append(e)
    if round(e, 0) == 2:
        price_Israel__2.append(e)
    if round(e, 0) == 3:
        price_Israel__3.append(e)
    if round(e, 0) == 4:
        price_Israel__4.append(e)
    if round(e, 0) == 5:
        price_Israel__5.append(e)
    if round(e, 0) == 6:
        price_Israel__6.append(e)
    if round(e, 0) == 7:
        price_Israel__7.append(e)
    if round(e, 0) == 8:
        price_Israel__8.append(e)
    if round(e, 0) == 9:
        price_Israel__9.append(e)

        
# Append to appropriate List

for e in unit_price_Italy_:
    if round(e, 0) == 1:
        price_Italy__1.append(e)
    if round(e, 0) == 2:
        price_Italy__2.append(e)
    if round(e, 0) == 3:
        price_Italy__3.append(e)
    if round(e, 0) == 4:
        price_Italy__4.append(e)
    if round(e, 0) == 5:
        price_Italy__5.append(e)
    if round(e, 0) == 6:
        price_Italy__6.append(e)
    if round(e, 0) == 7:
        price_Italy__7.append(e)
    if round(e, 0) == 8:
        price_Italy__8.append(e)
    if round(e, 0) == 9:
        price_Italy__9.append(e)
        
# Append to appropriate List

for e in unit_price_Singapore_:
    if round(e, 0) == 1:
        price_Singapore__1.append(e)
    if round(e, 0) == 2:
        price_Singapore__2.append(e)
    if round(e, 0) == 3:
        price_Singapore__3.append(e)
    if round(e, 0) == 4:
        price_Singapore__4.append(e)
    if round(e, 0) == 5:
        price_Singapore__5.append(e)
    if round(e, 0) == 6:
        price_Singapore__6.append(e)
    if round(e, 0) == 7:
        price_Singapore__7.append(e)
    if round(e, 0) == 8:
        price_Singapore__8.append(e)
    if round(e, 0) == 9:
        price_Singapore__9.append(e)
        
# Append totals to list

Italy_totals_list = [len(price_Italy__1),len(price_Italy__2), len(price_Italy__3), len(price_Italy__4),
               len(price_Italy__5), len(price_Italy__6), len(price_Italy__7), len(price_Italy__8),
               len(price_Italy__9)]

# Append totals to list

Israel_totals_list = [len(price_Israel__1),len(price_Israel__2), len(price_Israel__3), len(price_Israel__4),
               len(price_Israel__5), len(price_Israel__6), len(price_Israel__7), len(price_Israel__8),
               len(price_Israel__9)]


# Append totals to list

Singapore_totals_list = [len(price_Singapore__1),len(price_Singapore__2), len(price_Singapore__3), len(price_Singapore__4),
               len(price_Singapore__5), len(price_Singapore__6), len(price_Singapore__7), len(price_Singapore__8),
               len(price_Singapore__9)]
#sum list of totals

Italy_total_digits = 0

for e in Italy_totals_list:
    Italy_total_digits = Italy_total_digits + e
    
#sum list of totals

Israel_total_digits = 0

for e  in Israel_totals_list:
    Israel_total_digits = Israel_total_digits + e
      

#sum list of totals

Singapore_total_digits = 0

for e  in Singapore_totals_list:
    Singapore_total_digits = Singapore_total_digits + e

# Calculate Distribution List

Italy_dist_list = [(len(price_Italy__1)/Italy_total_digits),(len(price_Italy__2)/Italy_total_digits),
             (len(price_Italy__3)/Italy_total_digits), (len(price_Italy__4)/Italy_total_digits),
             (len(price_Italy__5)/Italy_total_digits), (len(price_Italy__6)/Italy_total_digits), 
             (len(price_Italy__7)/Italy_total_digits), (len(price_Italy__8)/Italy_total_digits),
             (len(price_Italy__9)/Italy_total_digits)]

print("Italy Distribution", Italy_dist_list)


# Calculate Distribution List

Singapore_dist_list = [(len(price_Singapore__1)/Singapore_total_digits),(len(price_Singapore__2)/Singapore_total_digits),
             (len(price_Singapore__3)/Singapore_total_digits), (len(price_Singapore__4)/Singapore_total_digits),
             (len(price_Singapore__5)/Singapore_total_digits), (len(price_Singapore__6)/Singapore_total_digits), 
             (len(price_Singapore__7)/Singapore_total_digits), (len(price_Singapore__8)/Singapore_total_digits),
             (len(price_Singapore__9)/Singapore_total_digits)]

print("Singapore Distribution", Singapore_dist_list)

# Calculate Distribution List

Israel_dist_list = [(len(price_Israel__1)/Israel_total_digits),(len(price_Israel__2)/Israel_total_digits),
             (len(price_Israel__3)/Israel_total_digits), (len(price_Israel__4)/Israel_total_digits),
             (len(price_Israel__5)/Israel_total_digits), (len(price_Israel__6)/Israel_total_digits), 
             (len(price_Israel__7)/Israel_total_digits), (len(price_Israel__8)/Israel_total_digits),
             (len(price_Israel__9)/Israel_total_digits)]

print("Israel Distribution", Israel_dist_list)

Italy_dist_list2 = [25.71, 33.92, 17.14, 6.07, 6.07, 3.21, 1.07, 6.42, .35]
Singapore_dist_list2 = [34.40, 24.73, 15.05, 11.82, 1.07, 3.22, 5.37, 4.30, 0]
Israel_dist_list2 = [18.91, 21.62, 13.51, 16.21, 8.10, 0.0, 5.40, 16.21, 0]


# Create a dataframe of the distributions

Italy_dist = pd.DataFrame(Italy_dist_list2, index =[1,2,3,4,5,6,7,8,9],
                                             columns =['Italy']) 

Italy_dist ['Bernford'] = bernford
Italy_dist ['Equal Weight'] = equal_weight


# Calculate error columns
 
Italy_dist['Equal Error'] = abs(Italy_dist['Italy'] - Italy_dist['Equal Weight'])/Italy_dist['Italy']

Italy_dist['Bernford Error'] = abs(Italy_dist['Italy'] - Italy_dist['Bernford'])/Italy_dist['Italy']
  

print(Italy_dist)


# Create a dataframe of the distributions

Israel_dist = pd.DataFrame(Israel_dist_list2, index =[1,2,3,4,5,6,7,8,9], 
                                              columns =['Israel']) 
Israel_dist ['Bernford'] = bernford
Israel_dist ['Equal Weight'] = equal_weight


# Calculate error columns
 
Israel_dist['Equal Error'] = abs(Israel_dist['Israel'] - Israel_dist['Equal Weight'])/Israel_dist['Israel']

Israel_dist['Bernford Error'] = abs(Israel_dist['Israel'] - Israel_dist['Bernford'])/Israel_dist['Israel']
  

print(Israel_dist)

# Create a dataframe of the distributions

Singapore_dist = pd.DataFrame(Singapore_dist_list2, index =[1,2,3,4,5,6,7,8,9], 
                                              columns =['Singapore']) 
Singapore_dist ['Bernford'] = bernford
Singapore_dist ['Equal Weight'] = equal_weight


# Calculate error columns
 
Singapore_dist['Equal Error'] = abs(Singapore_dist['Singapore'] - Singapore_dist['Equal Weight'])/Singapore_dist['Singapore']

Singapore_dist['Bernford Error'] = abs(Singapore_dist['Singapore'] - Singapore_dist['Bernford'])/Singapore_dist['Singapore']
  

print(Singapore_dist)

# Calculate root mean squared

Singapore_dist['Equal Error Squared'] = (Singapore_dist['Singapore'] - Singapore_dist['Equal Weight'])**2
Singapore_dist['Bernford Error Squared'] = (Singapore_dist['Singapore'] - Singapore_dist['Bernford'])**2

Singapore_bernford_sq = Singapore_dist['Bernford Error Squared']
Singapore_equal_sq = Singapore_dist['Equal Error Squared']

Singapore_sum_bernford_sq = 0
Singapore_sum_equal_sq = 0

for e in Singapore_bernford_sq:
    Singapore_sum_bernford_sq = Singapore_sum_bernford_sq + e
    
Singapore_Rt_mn_sq_bernford = math.sqrt(Singapore_sum_bernford_sq/9)
    
print("Singapore RMSE bernford is", Singapore_Rt_mn_sq_bernford)

for e in Singapore_equal_sq:
    Singapore_sum_equal_sq = Singapore_sum_equal_sq + e
    
Singapore_Rt_mn_sq_equal = math.sqrt(Singapore_sum_equal_sq/9)

print("Singapore RMSE equal is", Singapore_Rt_mn_sq_equal)


# Calculate root mean squared

Israel_dist['Equal Error Squared'] = (Israel_dist['Israel'] - Israel_dist['Equal Weight'])**2
Israel_dist['Bernford Error Squared'] = (Israel_dist['Israel'] - Israel_dist['Bernford'])**2

Israel_bernford_sq = Israel_dist['Bernford Error Squared']
Israel_equal_sq = Israel_dist['Equal Error Squared']

Israel_sum_bernford_sq = 0
Israel_sum_equal_sq = 0

for e in Israel_bernford_sq:
    Israel_sum_bernford_sq = Israel_sum_bernford_sq + e
    
Israel_Rt_mn_sq_bernford = math.sqrt(Israel_sum_bernford_sq/9)
    
print("Israel RMSE bernford is", Israel_Rt_mn_sq_bernford)

for e in Israel_equal_sq:
    Israel_sum_equal_sq = Israel_sum_equal_sq + e
    
Israel_Rt_mn_sq_equal = math.sqrt(Israel_sum_equal_sq/9)

print("Israel RMSE equal is", Israel_Rt_mn_sq_equal)



# Calculate root mean squared

Italy_dist['Equal Error Squared'] = (Italy_dist['Italy'] - Italy_dist['Equal Weight'])**2
Italy_dist['Bernford Error Squared'] = (Italy_dist['Italy'] - Italy_dist['Bernford'])**2

Italy_bernford_sq = Italy_dist['Bernford Error Squared']
Italy_equal_sq = Italy_dist['Equal Error Squared']

Italy_sum_bernford_sq = 0
Italy_sum_equal_sq = 0

for e in Italy_bernford_sq:
    Italy_sum_bernford_sq = Italy_sum_bernford_sq + e
    
Italy_Rt_mn_sq_bernford = math.sqrt(Italy_sum_bernford_sq/9)
    
print("Italy RMSE bernford is", Italy_Rt_mn_sq_bernford)

for e in Italy_equal_sq:
    Italy_sum_equal_sq = Italy_sum_equal_sq + e
    
Italy_Rt_mn_sq_equal = math.sqrt(Italy_sum_equal_sq/9)

print("Italy RMSE equal is", Italy_Rt_mn_sq_equal)
