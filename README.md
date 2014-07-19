LDA
===
Implementation of Linear Discriminant Analysis (LDA) algorithm which can be used for binary classification of multivariate data.
###Basic LDA (for 2 classes):
```python
>>> from LDA import LDA
>>> t = LDA([[2.4, 5.4, 3.3], [5, 6.6, 3.4], [2.1, 2.4, 2.5], [5, 3.1, 6]], ['A', 'A', 'B', 'B'])
>>> t.predict([2.3, 3.2, 2.4])
>>> {'A': 2.1366024891345905, 'B': 3.2231442139356759}
```
Each of the binary classes above ('A' and 'B') are predicted based on the standard discriminant function i.e. ![alt text](https://github.com/saifuddin778/LDA/raw/master/images/lda.png ""), and one with maximum output is the predicted class for the given input.
###Multiclass LDA (Pairwise implementation):
If the dataset has more than two classes i.e. K > 2, then it can be processed with all the possible combinations of classes, which results in ```K(K-1)/2``` classifiers trained for each combination. Moreover, the classification decision is made by the popular desicion system as specified by [Wu et, al. (2004)](http://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf):
```python
>>> from LDA import multiclass_LDA
>>> t = multiclass_LDA([[2.4, 5.4, 3.3], [5, 6.6, 3.4], [2.1, 2.4, 2.5], [5, 3.1, 6]], ['A', 'B', 'C', 'D'])
>>> t.predict([2.3, 3.2, 2.4])
>>> [{'A': score, 'B': score}, {'C': score, 'A': score}, {'B': score, 'C': score}]
```
