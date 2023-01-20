# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 18:35:43 2023

@author: Lenovo
"""
#import pacakages from different modules 
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmets
import matplotlib.pyplot as plt
import scipy.optimize as so
#read csv files for clustering 
data= pd.read_csv("co2emissions.csv")
print("\nemissions: \n", data)
#dropping the columns which are not needed
data= data.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
print("\nNew Population: \n", data)
#filling nan values 
data =data.fillna(0)
print("\nNew Population after filling null values: \n", data)
#tanspose the data
data= pd.DataFrame.transpose(data)
print("\nTransposed Dataframe: \n",data)
"""This function is use for get information of dataframe.Moreover we calculat count of 
NaN value,Once NaN value found drop NaN value or 
null value from a dataframe. Assign a cleaned data
 to cleanDataframe variable and  return a cleaned data frame.
        dataframe: Dataframe as parameter.
        cleandDataframe: Variable which has NaN value removed and return to function."""
def dataCleaning(data):
    # Information about dataframe
    print("Information of Transpose Data: ", data.info())
    # Calculate Null or NaN value into dataset
    print('Total number of NaN count in dataset: ', data.isna().sum())
    # Drop all NaN value and assign new dataframe.
    cleanDataframe = data.dropna()
    return cleanDataframe

data=dataCleaning(data)

#populating header with header information
data_header = data.iloc[0].values.tolist()
data.columns = data_header
print("/Header: \n",data)

#select the rows which are needed to be ploted 
data= data.iloc[2:]
print("\nNew Transposed Dataframe: \n",data)
data_ex = data[["Colombia","Argentina"]].copy()

def norm(array):
 """ Returns array normalised to [0,1]. Array can be a numpy array
 or a column of a dataframe"""
max_val = data_ex.max()
min_val = data_ex.min()
data_ex = (data_ex - min_val) / (max_val - min_val)
print("\nNew selected columns dataframe: \n", data_ex)

#plotting 5 clusters 
ncluster = 5
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(data_ex)

# extract labels and cluster centres
labels = kmeans.labels_

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)

# calculate the silhoutte score
print(skmets.silhouette_score(data_ex, labels))

# #Individual colours can be assigned to symbols. The label l is used to the␣
#↪→select the
# l-th number from the colour table.
plt.figure(figsize=(10.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
"tab:pink", "tab:gray", "tab:olive", "tab:cyan"] 
    
for l in range(ncluster): # loop over the different labels
    plt.plot(data_ex[labels==l]["Colombia"], data_ex[labels==l]["Argentina"], marker="o", markersize=3, color=col[l])    
# show cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]  
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Colombia")
plt.ylabel("Argentina")
plt.show() 

print(cen)

df_cen = pd.DataFrame(cen, columns=["Colombia", "Argentina"])
print(df_cen)
df_cen = df_cen * (max_val - min_val) + max_val
data_ex = data_ex * (max_val - min_val) + max_val
# print(df_ex.min(), df_ex.max())

print(df_cen)

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
"tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(ncluster): # loop over the different labels
    plt.plot(data_ex[labels==l]["Colombia"], data_ex[labels==l]["Argentina"], "o", markersize=10, color=col[l]) 
# show cluster centres
plt.plot(df_cen["Colombia"], df_cen["Argentina"], "dk", markersize=20)
plt.xlabel("Colombia")
plt.ylabel("Argentina")
plt.title(" fig ->5 Clusters of co2 emission of   Colombia and Argentina")
plt.show()
print(cen) 
""""
------------------------------------Curve Fitting------------------------------"""
 #this readfile function is used to read a file from the directory of the pc
def readFile(x):
    n_emiss= pd.read_csv("no2 emissions.csv");
    n_emiss= pd.read_csv(x)
    n_emiss= n_emiss.fillna(0.0)
    return n_emiss
n_emiss= readFile("no2 emissions.csv")
#transpose th data 
n_emiss=n_emiss.transpose()
print("\ntransposed data : \n", n_emiss)
'''
    This function is use for get information of dataframe.Moreover we calculat count of NaN 
    value,Once NaN value found 
    drop NaN value or null value from a dataframe. Assign a cleaned data to cleanDataframe variable and 
    return a cleaned data frame.
    dataframe: Dataframe as parameter.
    cleandDataframe: Variable which has NaN value removed and return to function.
'''
def dataCleaning(n_emiss):
    '''
        This function is use for get information of dataframe.Moreover we calculat count of NaN value,Once NaN value found 
        drop NaN value or null value from a dataframe. Assign a cleaned data to cleanDataframe variable and 
        return a cleaned data frame.
        dataframe: Dataframe as parameter.
        cleandDataframe: Variable which has NaN value removed and return to function.
    '''
    # Information about dataframe
    print("Information of Transpose Data: ", n_emiss.info())

    # Calculate Null or NaN value into dataset
    print('Total number of NaN count in dataset: ', n_emiss.isna().sum())

    # Drop all NaN value and assign new dataframe.
    cleanDataframe = n_emiss.dropna()
    return cleanDataframe
n_emiss=dataCleaning(n_emiss)
n_emiss= pd.DataFrame(n_emiss)

header3 = n_emiss.iloc[0].values.tolist()
n_emiss.columns = header3
print("\nemissions  Header: \n",n_emiss)
n_emiss= n_emiss["United Kingdom"]
print("\nemission after dropping columns: \n",n_emiss)

n_emiss.columns = ["emission"]
print("\nemission: \n",n_emiss)

n_emiss= n_emiss.iloc[5:]
n_emiss= n_emiss.iloc[:-1]
print("\nemission: \n",n_emiss)

n_emiss= n_emiss.reset_index()
print("\changed index: \n",n_emiss)

n_emiss= n_emiss.rename(columns={"index": "Year", "United Kingdom": "emission"})
print("\nrenamed columns: \n",n_emiss)
#Define the exponential function and the logistics functions for fitting.
def exponential(s, q0, h):
    s= s - 1978.0
    x = q0 * np.exp(h*s)
    return x
print(type(n_emiss["Year"].iloc[1]))
n_emiss["Year"] = pd.to_numeric(n_emiss["Year"])
print("\nemission: \n", type(n_emiss["Year"].iloc[8]))
param, covar = so.curve_fit(exponential, n_emiss["Year"], n_emiss["emission"],
p0=(4.978423, 0.03))
# fit exponential growth
n_emiss["fit"] = exponential(n_emiss["Year"], *param)
n_emiss.plot("Year", ["emission", "fit"], label=["new emissions of nitrous oxide","new curve fit"])
plt.legend()
plt.show()
# predict fit for future years
year = np.arange(1960, 2040)

print("\nForecast Years: \n", year)

forecast = exponential(year, *param)

plt.figure()

plt.plot(n_emiss["Year"], n_emiss["emission"], label="forecasted emission ")

plt.plot(year, forecast, label="Forecast")

plt.xlabel("Year")

plt.ylabel("forecasted fit ")

plt.title("emissions of no2 to be predicted ")

plt.legend()

plt.show()

# err_ranges function
def err_ranges(x, exponential, param, sigma):
    import itertools as iter
    # initiate arrays for lower and upper limits
    lower = exponential(x, *param)
    upper = lower
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
    for p in pmix:
        y = exponential(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        print("\nLower: \n", lower)
        print("\nUpper: \n", upper)        
    return lower, upper






