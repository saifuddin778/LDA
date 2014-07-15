import sys
sys.dont_write_bytecode = True

"""
Testing LDA
"""
def test_LDA():
    import copy
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
    

test_LDA()
    
