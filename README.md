LDA
===
Implementation of Linear Discriminant Analysis (LDA) algorithm which can be used for binary classification of multivariate data.

```python
>>> from LDA import LDA
>>> t = LDA([[2.4, 5.4, 3.3], [5, 6.6, 3.4], [2.1, 2.4, 2.5], [5, 3.1, 6]], ['A', 'A', 'B', 'B'])
>>> t.predict([2.3, 3.2, 2.4])
>>> ...
```

Predicts based on the standard discriminant function i.e. ![alt text](https://github.com/saifuddin778/LDA/raw/master/images/lda.png "")
