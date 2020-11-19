# klasifikasi jenis kendaraan dengan 
# Decision Tree

from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# model untuk ketiga classifier
cDT = tree.DecisionTreeClassifier()
cSVM = svm.SVC()
cKNN = neighbors.KNeighborsClassifier()
cNB = GaussianNB()

# data latih
# [pintu, roda,    cc]
X = [[2,    4,   1500], 
     [5,    4,   1000], 
     [2,    4,   2500], 
     [4,    4,   2000], 
     [3,    8,   3000],
     [2,    6,   4000], 
     [2,   12,   4100],
     [2,    6,   5500],
     [4,    5,   3200],
     [3,    3,   500]]

Y = ['lambor', 'avanza', 'bmwz4', 'mercedez', 'bus', 'truck', 'kontener', 'pickup', 'fortuner', 'bajay']

# latih classifier
cDT = cDT.fit(X, Y)
cSVM = cSVM.fit(X, Y)
cKNN = cKNN.fit(X, Y)
cNB = cNB.fit(X, Y)

# prediksi data test
Y_DT = cDT.predict(X)
Y_SVM = cSVM.predict(X)
Y_KNN = cKNN.predict(X)
Y_NB = cNB.predict(X)


# print akurasi
print("Akurasi Decision Tree : ", accuracy_score(Y, Y_DT))
print("Akurasi SVM : ", accuracy_score(Y, Y_SVM))
print("Akurasi KNN : ", accuracy_score(Y, Y_KNN))
print("Akurasi Naive Bayes : ", accuracy_score(Y, Y_NB))
