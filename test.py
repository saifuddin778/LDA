from __future__ import division
import sys
import copy
sys.dont_write_bytecode = True

"""
Testing LDA
"""
def test_LDA():
    from LDA import LDA
    x = [
        [2.95, 6.63],
        [2.53, 7.79],
        [3.57, 5.65],
        [3.16, 5.47],
        [2.58, 4.46],
        [2.16, 6.22],
        [3.27, 3.52]
    ]
    e = copy.deepcopy(x)
    y = [1,1,1,1,2,2,2]
    t = LDA(x, y)
    for a in e:
        r = t.predict(a)
        print max(r, key=r.get)
    

"""
Testing multiclass LDA
"""
def test_multiclass_LDA():
    from LDA import multiclass_LDA
    from sklearn import datasets
    print 'data loaded..'
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']
    l = copy.deepcopy(x)
    m = copy.deepcopy(y)
    t = multiclass_LDA(x, y)
    for a,b in zip(l, m):
        print t.predict(a), b
    

#t = test_LDA()
#t  = test_multiclass_LDA()

if __name__ == '__main__' and len(sys.argv) == 2:
    print sys.argv
    method_to_test = sys.argv[1]
    if method_to_test == 'LDA':
        test_LDA()
    elif method_to_test == 'multiclass_LDA':
        test_multiclass_LDA()
