#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[2]:


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


# This repo contains an introduction to [Jupyter](https://jupyter.org) and [IPython](https://ipython.org).
# 
# Outline of some basics:
# 
# * [Notebook Basics](../examples/Notebook/Notebook%20Basics.ipynb)
# * [IPython - beyond plain python](../examples/IPython%20Kernel/Beyond%20Plain%20Python.ipynb)
# * [Markdown Cells](../examples/Notebook/Working%20With%20Markdown%20Cells.ipynb)
# * [Rich Display System](../examples/IPython%20Kernel/Rich%20Output.ipynb)
# * [Custom Display logic](../examples/IPython%20Kernel/Custom%20Display%20Logic.ipynb)
# * [Running a Secure Public Notebook Server](../examples/Notebook/Running%20the%20Notebook%20Server.ipynb#Securing-the-notebook-server)
# * [How Jupyter works](../examples/Notebook/Multiple%20Languages%2C%20Frontends.ipynb) to run code in different languages.

# You can also get this tutorial and run it on your laptop:
# 
#     git clone https://github.com/ipython/ipython-in-depth
# 
# Install IPython and Jupyter:
# 
# with [conda](https://www.anaconda.com/download):
# 
#     conda install ipython jupyter
# 
# with pip:
# 
#     # first, always upgrade pip!
#     pip install --upgrade pip
#     pip install --upgrade ipython jupyter
# 
# Start the notebook in the tutorial directory:
# 
#     cd ipython-in-depth
#     jupyter notebook
