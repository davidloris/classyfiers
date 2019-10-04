import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm,ensemble, tree, neighbors
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#Reading the Train.CSV file. This file contains the entire features of the project but it is not certain 
# that inserting them all brings a better result in learning. For this reason there are other reading functions with less features


#data = pd.read_csv('train.csv')
data = pd.read_csv('train.csv')[["Gender","Eye color","Race","Hair color","Height","Publisher","Skin color","Alignment","Weight","Agility","Accelerated Healing","Lantern Power Ring","Dimensional Awareness","Cold Resistance","Durability","Stealth","Energy Absorption","Flight","Danger Sense"]]
#data = pd.read_csv('train.csv')[["Gender","Race","Height","Publisher","Alignment","Weight","Agility","Accelerated Healing","Lantern Power Ring","Dimensional Awareness","Cold Resistance","Durability","Stealth","Energy Absorption","Flight","Danger Sense"]]
#data = pd.read_csv('train.csv')[["Race","Hair color","Height","Skin color","Alignment","Weight","Agility","Accelerated Healing","Lantern Power Ring","Dimensional Awareness","Cold Resistance","Durability","Stealth","Energy Absorption","Flight","Danger Sense"]]

# Method that is used to clean the data of our Dataset

def clean_data(data):
    
    # With this command we transform all True False values to 0 or 1
    data = data * 1
    
    # Delete the Nan fields in the "Gender" column
    data['Gender'].map({'Female': 0, 'Male': 1})
    data['Gender'].fillna(data['Gender'].mean())
    
    # I put to Nan all the negative values of height and weight. Then I replace the value
    # Nan with the media
    data["Height"][data["Height"] < 0] = np.nan
    data["Height"].fillna(data["Height"].mean(), inplace=True)

    data["Weight"][data["Weight"] < 0] = np.nan
    data["Weight"].fillna(data["Weight"].mean(), inplace=True)
    
    #Errors in the names of various colors are corrected
    data["Eye color"] = np.where(data["Eye color"] == 'bown', "brown", data["Eye color"])
    data["Hair color"] = np.where(data["Hair color"] == 'black', "Black", data["Hair color"])
    return data


#method that insert the various values of the column on the columns
def split_columns(name_column):
    df = pd.DataFrame(data[name_column])
    df = df.replace(np.nan, 'Prova')
    df_pivot = df.pivot(columns=name_column, values=name_column)
    df_pivot = df_pivot.replace(np.nan, 0)
    df_pivot2 = df_pivot.where(df_pivot['Prova'] != 'Prova')
    col = df_pivot2.columns
    def f(x):
        if type(x) == str:
            return 1
        else: 
            return x

    for i in col:
        df_pivot2[i] = df_pivot2[i].apply(f)

    df_pivot2.drop('Prova', axis=1, inplace = True)
    
    df_pivot2.fillna(df_pivot2.mean(), inplace=True)
    
    data.drop(name_column, axis=1, inplace = True)
    
    data[df_pivot2.columns] = df_pivot2
    
    return data

#I clean our Dataset with the method previously created
data = clean_data(data)

#I take all the columns with the values ​​to Nan
missing_values_index = missing_values_count[np.where(missing_values_count > 0, True, False)].index

#For each value containing Nan data I use the "split_columns" method
for i in missing_values_index:
    split_columns(i)

#I insert my Label inside, inside the X I insert the remaining table  

y = data["Alignment"]

X = data.drop(columns=['Alignment'])

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)

X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaler, y_train, test_size=0.3, random_state=42)

X_train_scaler_final = scaler.fit_transform(X_train_final)
X_test_scaler_final = scaler.fit_transform(X_val)

#APPLY THE FIRST METHOD
clf = svm.SVC(C = 10.0, gamma = 1e-5, random_state = 42)
clf.fit(X_train_scaler_final, y_train_final) 
yhat = clf.predict(X_test_scaler_final)

acc_val1 = metrics.accuracy_score(y_val,yhat)
print("VALORE FINALE METODO 1:",  acc_val1) 

#APPLY THE SECOND METHOD
clf = svm.SVC(C=100.0,gamma = 1e-5,random_state=42)
clf.fit(X_train_scaler_final, y_train_final) 
yhat = clf.predict(X_test_scaler_final)

acc_val_2 = metrics.accuracy_score(y_val,yhat)
print("VALORE FINALE METODO 2:", acc_val_2)

#APPLY THE THIRD METHOD
clf = svm.SVC(C=1000.0,gamma = 1e-6,random_state=42)
clf.fit(X_train_scaler_final, y_train_final) 
yhat = clf.predict(X_test_scaler_final)

acc_val_3 = metrics.accuracy_score(y_val,yhat)
print("VALORE METODO 3:",acc_val_3)

#APPLY THE FOURTH METHOD
clf = ensemble.GradientBoostingClassifier(random_state=42)
clf.fit(X_train_scaler_final, y_train_final) 
yhat = clf.predict(X_test_scaler_final)

acc_val_4 = metrics.accuracy_score(y_val,yhat)
print("VALORE METODO 4:",acc_val_4)

#APPLY THE FIFTH METHOD
clf = tree.DecisionTreeClassifier()
clf.fit(X_train_scaler_final, y_train_final) 
yhat = clf.predict(X_test_scaler_final)

acc_val_4 = metrics.accuracy_score(y_val,yhat)
print("VALORE METODO 5:",acc_val_4)

#APPLY THE SIXTH METHOD
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train_scaler_final, y_train_final) 
yhat = clf.predict(X_test_scaler_final)

acc_val_4 = metrics.accuracy_score(y_val,yhat)
print("VALORE METODO 6:",acc_val_4)

#APPLY THE SEVENTH METHOD
clf = neighbors.KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train_scaler_final, y_train_final) 
yhat = clf.predict(X_test_scaler_final)

acc_val_4 = metrics.accuracy_score(y_val,yhat)
print("VALORE METODO 7:",acc_val_4)

#################METHOD Naive-Bayes#############################

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gnb = GaussianNB()
pred = gnb.fit(X_train_scaler_final, y_train_final).predict(X_test_scaler_final)
print("VALORE Naive-Bayes: ",accuracy_score(y_val, pred, normalize = True))

#############################################


#################METHOD LinearSVC#############################

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

svc_model = LinearSVC(random_state=42)
#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(X_train_scaler_final, y_train_final).predict(X_test_scaler_final)
#print the accuracy score of the model
print("VALORE LinearSVC: ",accuracy_score(y_val, pred, normalize = True))


#############################################


#############METHOD RandomForest#########################
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=3)    # n_estimators is number of Decision Tree in this algorithm.
rf.fit(X_train_scaler_final, y_train_final.ravel())
rf.predict(X_test_scaler_final)
print("VALORE Random Forest: ",accuracy_score(y_val, pred, normalize = True))


#######################################


##################METHOD LogisticRegression######################
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaler_final, y_train_final)
model.predict(X_test_scaler_final)
print("VALORE LogisticRegression: ",accuracy_score(y_val, pred, normalize = True))
########################################

#IN THIS PART DAVO THEN TO VALIDATE THE VARIOUS MODELS THAT HAVE TAKEN A GREATER RESULT
print("############### Validation #############")
#clf = ensemble.GradientBoostingClassifier(random_state=42)
#clf.fit(X_train_scaler, y_train) 
#yhat = clf.predict(X_test_scaler)

#acc3 = metrics.accuracy_score(y_test,yhat)
#print("VALORE FINALE METODO 4:",acc3)

#clf = neighbors.KNeighborsClassifier(n_neighbors=7)
#clf.fit(X_train_scaler, y_train) 
#yhat = clf.predict(X_test_scaler)

#acc3 = metrics.accuracy_score(y_test,yhat)
#print("VALORE FINALE METODO 7:",acc3)


rf = RandomForestClassifier(n_estimators=100, random_state=3)    # n_estimators is number of Decision Tree in this algorithm.
rf.fit(X_train_scaler, y_train.ravel())
pred = rf.predict(X_test_scaler)
print("VALORE FINALE RANDOM FOREST: ",accuracy_score(y_test, pred, normalize = True))



#model = LogisticRegression()
#model.fit(X_train_scaler, y_train)
#pred = model.predict(X_test_scaler)
#print("LogisticRegression Algorithm test accuracy: ",accuracy_score(y_test, pred, normalize = True))



#svc_model = LinearSVC(random_state=42)
#pred = svc_model.fit(X_train_scaler, y_train).predict(X_test_scaler)
#print("LinearSVC accuracy : ",accuracy_score(y_test, pred, normalize = True))
