# Importing FILEE
%matplotlib inline
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

import matplotlib
import matplotlib.pyplot as plt

# Reading the '.CSV' Files
# df=pd.read_csv(csv_filename,index_col=0)


df1=pd.read_csv("a1_raw.csv" , skiprows=[1,2,3,4])
df2=pd.read_csv("a1_va3.csv")
df3=pd.read_csv("a2_raw.csv" , skiprows=[1,2,3,4])
df4=pd.read_csv("a2_va3.csv")
df5=pd.read_csv("a3_raw.csv" , skiprows=[1,2,3,4])
df6=pd.read_csv("a3_va3.csv")
df7=pd.read_csv("b1_raw.csv" , skiprows=[1,2,3,4])
df8=pd.read_csv("b1_va3.csv")
df9=pd.read_csv("b3_raw.csv" , skiprows=[1,2,3,4])
df10=pd.read_csv("b3_va3.csv")
df11=pd.read_csv( "c1_raw.csv", skiprows=[1,2,3,4])
df12=pd.read_csv("c1_va3.csv")
df13=pd.read_csv("c3_raw.csv", skiprows=[1,2,3,4])
df14=pd.read_csv("c3_va3.csv")

# Removing the 'timestamp' & 'phase' labels from unprocessed files
df1.drop('timestamp',axis=1,inplace=True)
df1.drop('phase',axis=1,inplace=True)
df3.drop('timestamp',axis=1,inplace=True)
df3.drop('phase',axis=1,inplace=True)
df5.drop('timestamp',axis=1,inplace=True)
df5.drop('phase',axis=1,inplace=True)
df7.drop('timestamp',axis=1,inplace=True)
df7.drop('phase',axis=1,inplace=True)
df9.drop('timestamp',axis=1,inplace=True)
df9.drop('phase',axis=1,inplace=True)
df11.drop('timestamp',axis=1,inplace=True)
df11.drop('phase',axis=1,inplace=True)
df13.drop('timestamp',axis=1,inplace=True)
df13.drop('phase',axis=1,inplace=True)

# Visualising the Table 1
df1.head()
#df1.shape

# Visualising the Table 2
#df2.head()
#df2.shape
df2.Phase.unique()

# Visualising the Column Labels in the two tables.
#df1.columns
#df2.columns

# Renaming 'Phase' Column for convinience
df2.rename(columns={'Phase': 'phase'}, inplace=True)
df4.rename(columns={'Phase': 'phase'}, inplace=True)
df6.rename(columns={'Phase': 'phase'}, inplace=True)
df8.rename(columns={'Phase': 'phase'}, inplace=True)
df10.rename(columns={'Phase': 'phase'}, inplace=True)
df12.rename(columns={'Phase': 'phase'}, inplace=True)
df14.rename(columns={'Phase': 'phase'}, inplace=True)

#  Concatenating the Dataframes
p1 = pd.concat([df1,df2],axis=1)
p2 = pd.concat([df3,df4],axis=1)
p3 = pd.concat([df5,df6],axis=1)
p4 = pd.concat([df7,df8],axis=1)
p5 = pd.concat([df9,df10],axis=1)
p6 = pd.concat([df11,df12],axis=1)
p7 = pd.concat([df13,df14],axis=1)

df= pd.concat([p1,p2,p3,p4,p5,p6,p7])

# Encoding Phase Labels and Estimating number of instances of Differrent Labels
df.phase.unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['phase'] = le.fit_transform(df['phase'])
df.groupby('phase').count()
df.phase.unique()


# Randomising the Data before Splitting
# Before
df.head()
df = df.sample(frac=1)

#After
df.head()

# Extracting the Feautre & Label Vector + Splitting into Test & Train
cols = list(df.columns)
features = cols
features.remove('phase')

len(features)
X = df[features]
y = df['phase']
X.shape

# split dataset to 60% training and 40% testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4, random_state=0)

# Normalize the Dataset for Easier Parameter Selection
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

print(X_train.shape, y_train.shape)







# Classification- Supervised Learning Task
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from time import time
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score , classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

# Apply PCA with the same number of dimensions as variables in the dataset
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
#pca = PCA(n_components=18)
pca.fit(X)

# First we reduce the data to two dimensions using PCA to capture variation
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)
#print(reduced_data[:10])  # print upto 10 elements

# split dataset to 60% training and 40% testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4, random_state=0)

print (X_train.shape, y_train.shape,X_test.shape, y_test.shape)

# Cross Validation Parameter
# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler


# Scale
X_train_scale =scale(X_train)
X_test_scale  =scale(X_test)

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
obj = LogisticRegression(max_iter=100000)
obj.fit(X=X_train,y=y_train)
Y_pred_cv = obj.predict(X_test)
confusion_matrix(y_true=y_test,y_pred=Y_pred_cv)
print(classification_report(y_true = y_test,y_pred=Y_pred_cv))

#  Decision Tree accuracy and time elapsed caculation
dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
clf_dt1= dt.fit(X_train,y_train)

print ("Acurracy: ", clf_dt1.score(X_test,y_test))



# Random Forest accuracy and time elapsed caculation
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf1 = rf.fit(X_train,y_train)
print ("Acurracy: ", clf_rf1.score(X_test,y_test))

# Naive Bayes accuracy and time elapsed caculation
nb = BernoulliNB()
print ("NaiveBayes")
clf_nb=nb.fit(X_train,y_train)
print ("Acurracy: ", clf_nb.score(X_test,y_test))




# SVM Accuracy
# SVM with a Linear Kernel
print ("SVM")

svc = SVC()
clf_svc=svc.fit(X_train_minmax, y_train)
print ("Acurracy: ", clf_svc.score(X_test_minmax,y_test) )
