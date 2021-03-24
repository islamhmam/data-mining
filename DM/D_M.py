import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_excel('Juice_quality.xlsx')

X_data = data.drop('quality',axis=1)

y_class = data['quality']

 #split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_data, y_class, test_size = 0.20)

#Dicsion tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Naive Bays
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_pred))


# KNN
model = KNeighborsClassifier(n_neighbors=9).fit(X_train,y_train)
y_pred= model.predict(X_test)
print(classification_report(y_test, y_pred))