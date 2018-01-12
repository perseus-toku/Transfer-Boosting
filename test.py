from adaboost import AdaBoost
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from scipy import sparse
from sklearn import ensemble

import TransferBoosting
"""

sci_categories = [ "sci.crypt","sci.electronics", "sci.med", "sci.space"]
rec_categories = ["rec.autos",  "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"]
rec_train = fetch_20newsgroups(subset='train',categories = rec_categories, shuffle = False, random_state = 42)
rec_test = fetch_20newsgroups(subset='test',categories = rec_categories, shuffle = False, random_state = 42)
sci_train = fetch_20newsgroups(subset='train',categories = sci_categories, shuffle = True, random_state = 42)
sci_test  = fetch_20newsgroups(subset='test',categories = sci_categories, shuffle = True, random_state = 42)
rec_train = rec_train.data +rec_test.data
rec_train = rec_train
sci_train = sci_train.data[:500]
rec_target = np.ones(len(rec_train))
sci_target = np.zeros(500)
sci_test_target = np.zeros(len(sci_test.data))
tfidf_transformer = TfidfTransformer(use_idf=True)
count_vect = CountVectorizer()
combinded_train = rec_train + sci_train
combined_counts = count_vect.fit_transform(combinded_train)
combined_tf = tfidf_transformer.fit_transform(combined_counts)
combined_target = np.concatenate((rec_target, sci_target), axis=0)
rec_counts = count_vect.transform(rec_train)
sci_counts = count_vect.transform(sci_train)
sci_test_counts = count_vect.transform(sci_test.data)
rec_train_tf = tfidf_transformer.transform(rec_counts)
sci_train_tf = tfidf_transformer.transform(sci_counts)
sci_test_tf = tfidf_transformer.transform(sci_test_counts)

#combined_target = np.asarray(combined_target)
#combined_target[combined_target==0] = -1
"""


#model = linear_model.LogisticRegression()
#model.fit(combined_tf.A, combined_target)
#preds= model.predict(sci_test_tf.A)
#c = np.sum([ x for x in preds == sci_test_target])
#print("base model error rate", c/len(sci_test_target))

X=[]
y=[]
test = []
test_y = []
with open ("train","r") as f:
    for line in f:
        lines= line.strip().split()
        x=[0]*123
        y.append(int(lines[0]))
        for i in range(1,len(lines)):
            cur = lines[i]
            ind= int(cur.strip().split(":")[0])
            v= float(cur.strip().split(":")[0])
            x[ind-1] = v
        X.append(x)

with open ("test","r") as fp:
    for line in fp:
        lines= line.strip().split()
        x=[0]*123
        test_y.append(int(lines[0]))
        for i in range(1,len(lines)):
            cur = lines[i]
            ind= int(cur.strip().split(":")[0])
            v= float(cur.strip().split(":")[0])
            x[ind-1] = v
        test.append(x)


#ada_model = AdaBoost(debug=True)
#ada_model.fit(X,y,M=10)
#ada_model.predict_and_calculate_error(test,test_y)


sources = [X[:400],X[400:800]]
sources_y = [y[:400],y[400:800]]
target = X[800:]
target_y = y[800:]

model = TransferBoosting.TransferBoost()
model.fit(sources, sources_y, target, target_y)
preds = model.predict_and_evaluate(test,test_y)
print(preds)


model.basic_model_evaluate(sources, sources_y, target, target_y,test,test_y)