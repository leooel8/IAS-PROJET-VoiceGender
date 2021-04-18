#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


# In[121]:


class RandomForest():
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 max_leaf_nodes=None,
                 min_samples_leaf=1, 
                 min_samples_split=2, 
                 random_state=None
                ):
        self.estimators = [DecisionTreeClassifier(max_depth=max_depth,
                                             max_leaf_nodes=max_leaf_nodes,
                                             min_samples_leaf=min_samples_leaf,
                                             min_samples_split=min_samples_split
                                            ) for i in range(n_estimators)
                          ]
        self.classes = []
    
    def fit(self, X, y):
        X1 = np.array(X.copy())
        y1 = np.array(y.copy())
        n = len(self.estimators)
        N = X1.shape[0]
        for (i, estimator) in enumerate(self.estimators):
            begin = int(i*N/n)
            end = int((i+1)*N/n) if i == n-1 else N
            X_  = X1[begin:end]
            y_ = y1[begin:end]
            estimator.fit(X_, y_)
            
        for i in y1:
            b = True
            for j in self.classes:
                b = b and i != j
                if not(b) : break
            if b : self.classes.append(i)
    
    def predict(self, X):
        X1 = np.array(X)
        predictions = []
        for i in range(X1.shape[0]):
            x = np.array([X1[i]])
            xpreds = [estimator.predict(x) for estimator in self.estimators]
            occs = [xpreds.count(k) for k in self.classes]
            predictions.append(self.classes[np.argmax(occs)])
        return np.array(predictions) if len(X.shape)>1 else predictions[0]
    
    def score(self, X, y):
        y1 = np.array(y)
        y_pred = self.predict(X)
        error = np.array([1 if y1[i] == y_pred[i] else 0 for i in range(y1.size)])
        return error.sum()/error.size
    

