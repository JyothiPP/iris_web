#importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle

#Loading the dataset
data=pd.read_excel("/workspaces/iris_web/iris .xls")

#Finding the outliers in 'SW'
Q1=np.percentile(data['SW'],25,method='midpoint')
Q2=np.percentile(data['SW'],50,method='midpoint')
Q3=np.percentile(data['SW'],75,method='midpoint')
IQR=Q3-Q1
low_lim=Q1-1.5*IQR
up_lim=Q3+1.5*IQR

outlier=[]
for i in data['SW']:
    if (i>up_lim)or(i<low_lim):
        outlier.append(i)


#Finding the index of the outliers
ind_1=data.loc[data['SW']<low_lim].index
ind_2=data.loc[data['SW']>up_lim].index

#Removing the outliers
data.drop(ind_1,inplace=True)
data.drop(ind_2,inplace=True)

#Defining independent and dependent variables
x=data.drop(['Classification'],axis=1)
y=data['Classification']

#Splitting the data in such a way that 75% data is for training and 25% is for testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)

#Logistic Regression
from sklearn import linear_model
lr=linear_model.LogisticRegression()
lr_model=lr.fit(x_train,y_train)
lr_pred=lr_model.predict(x_test)
lr_acc=accuracy_score(y_test,lr_pred)

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=4,metric='euclidean')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
knn_acc=accuracy_score(y_test,y_pred)

#SVM
from sklearn.svm import SVC
svm_cls=SVC(kernel='linear')
svm_cls=svm_cls.fit(x_train,y_train)
y_pred_svm=svm_cls.predict(x_test)
svm_acc=accuracy_score(y_test,y_pred_svm)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_cls=DecisionTreeClassifier()
dt_model=dt_cls.fit(x_train,y_train)
y_pred_dt=dt_model.predict(x_test)
dt_acc=accuracy_score(y_test,y_pred_dt)

#Random Tree
from sklearn.ensemble import RandomForestClassifier
rf_cls=RandomForestClassifier()
model_rf=rf_cls.fit(x_train,y_train)
y_pred_rf=model_rf.predict(x_test)
rf_acc=accuracy_score(y_test,y_pred_rf)

#Printing accuracy scores of models
print("Accuracy score of Logistic regression model :",lr_acc)
print("Accuracy score of KNN model :",knn_acc)
print("Accuracy score of SVM model :",svm_acc)
print("Accuracy score of Decision Tree model :",dt_acc)
print("Accuracy score of Random Forest model :",rf_acc)

#Creating pickle file for SVM model
pickle_file="svm_model.pickle"
with open(pickle_file,'wb') as file:
    pickle.dump(svm_cls,file)

print(f"Best model has been pickled and saved to {pickle_file}")