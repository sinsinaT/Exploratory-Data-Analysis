import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calmap
from pandas_profiling import ProfileReport

df = pd.read_csv("supermarket_sales.csv")
df.tail(10)
df.head()
df.columns()
df.info()
df.dtypes
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace = True)
df.describe()
sns.distplot(df["Rating"])

"""univariate analsysis """

plt.axvline(x=np.mean(df["Rating"]), c= "red", ls="--", label="mean"    )
plt.axvline(x=np.percentile(df["Rating"], 25), c="green", ls= "--",label="25-76")
plt.axvline(x=np.percentile(df["Rating"], 75), c="green", ls= "--")
plt.legend()

df.hist(figsize =(10,10))
sns.countplot(df["Branch"])
df["Branch"].value_counts()
sns.countplot(df["Payment"])

"""bivariate analsysis """
sns.scatterplot(df["Rating"], df["gross income"])
sns.regplot(df["Rating"], df["gross income"])
sns.boxplot(x=df["Branch"], y=df["gross income"])
sns.boxplot(x=df["Gender"], y=df["gross income"])
df.groupby(df.index).mean()
sns.lineplot(x= df.groupby(df.index).mean().index, y=df.groupby(df.index).mean()["gross income"])
sns.pairplot(df)

"""remove duplicate and missing values"""
df.duplicated().sum()
df[df.duplicated() == True]
df.drop_duplicates(inplace =True)

df.isna().sum()
sns.heatmap(df.isnull(),cbar = False)

df.fillna(df.mean(),inplace = True)
df.mode().iloc[0]
prof = ProfileReport(df)

""" correlation analysis"""
round(np.corrcoef(df["gross income"], df["Rating"])[1][0],2)
np.round(df.corr(),2)
sns.heatmap(np.round(df.corr(),2),annot=True)