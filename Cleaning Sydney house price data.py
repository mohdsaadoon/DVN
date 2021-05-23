# import packages
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure


pwd()

#read dataset 
house_df= pd.read_csv("C:\\Users\\Dell 990\\Desktop\\MDSI\\DVN\\AT3\\SydneyHousePrices.csv\\SydneyHousePrices.csv")

house_df.head()

#drop Id col
house_df.drop("Id",axis=1,inplace=True) 

print(house_df.shape)

house_df

house_df.describe()

house_df.isnull().sum()

house_df.dtypes

sns.pairplot(house_df)

corrmat = house_df.corr() 
  
f, ax = plt.subplots(figsize =(12, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 


def boxplt(house_df):
    for i in house_df.columns:
        if(house_df[i].dtype== np.int64 or house_df[i].dtype== np.float64):
            fig, ax =plt.subplots(figsize=(8,6))
            sns.boxplot(x = house_df[i])
boxplt(house_df)  




def find_outliers(house_df):
    for i in house_df.columns:
        if(house_df[i].dtype== np.int64):
            q1=house_df[i].quantile(0.25)
            q3=house_df[i].quantile(0.75)
            iqr=q3-q1
            upper_bound= q3 + 1.5 * iqr
            lower_bound= q1 - 1.5 * iqr
            for k in range(len(house_df[i])):
                if house_df[i].iloc[k]<lower_bound:
                    house_df[i].iloc[k]=lower_bound
                if house_df[i].iloc[k]>upper_bound:
                    house_df[i].iloc[k]=upper_bound

find_outliers(house_df)




#controling function of find_outlier 
def boxplt(house_df):
    for i in house_df.columns:
        if(house_df[i].dtype== np.int64 or house_df[i].dtype== np.float64):
            fig, ax =plt.subplots(figsize=(8,6))
            sns.boxplot(x = house_df[i])
boxplt(house_df)  




#percent of missing values
def percent_missing(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    gf=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    gf= gf[gf["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    fig=sns.barplot(gf.index, gf["Percent"],color="purple",alpha=0.4)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Percent of missing values', fontsize=16)
    plt.title('Missing values of feature', fontsize=16)
    return gf


percent_missing(house_df)

house_df

sns.heatmap(house_df.isnull(),yticklabels='auto',cmap= "viridis")

from sklearn.impute import SimpleImputer
columns = ['car','bed']
imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
house_df[columns] = imp_mean.fit_transform(house_df[columns])
house_df.isnull().sum()


house_df

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

house_df

house_df_2=house_df[(house_df.Date >= '2017-01-01') &(house_df.Date <= '2018-12-31')]
house_df_2=house_df_2[(house_df_2.propType == 'house')]


newhouse_df = house_df.apply(preprocessing.LabelEncoder().fit_transform)
newhouse_df.isnull().sum()

sns.jointplot(x= "bed",y= "sellPrice",data=newhouse_df ,kind= "reg")
