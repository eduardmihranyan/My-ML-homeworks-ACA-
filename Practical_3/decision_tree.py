import dtOmitted as dt
import numpy as np

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        # TODO: Build a tree that has self.max_depth
        if type(X) is np.ndarray:
            X=X.tolist()
        if type(Y) is np.ndarray:
            Y=Y.tolist()

        data=[X[i]+[Y[i]] for i in range(len(X))]


        self.root=dt.build_tree(data,max_depth=self.max_depth)

    def predict1(self,x, v):
        if v.is_leaf==True:
            return v.result
        else:
            xval=x[v.column]
            val=v.value
            if type(val) == int or type(val) == float:
                if xval>val:
                    return self.predict1(x,v.true_branch)
                else:
                    return self.predict1(x, v.false_branch)
            if type(val)==str:
                if xval==val:
                    return self.predict1(x,v.true_branch)
                else:
                    return self.predict1(x,v.false_branch)


    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        # TODO: Evaluate label of all the elements in `X` and
        # return same size list with labels.
        # TODO: Remove this toto and the todo above after you
        # implement the todo above.


        return [self.predict1(row,self.root) for row in X]

def main():
    X = [['slashdot', 'USA', 'yes', 18], ['google', 'France', 'yes', 23], ['reddit', 'USA', 'yes', 24]]
    Y = ['None', 'Premium', 'Basic']
    tree = DecisionTree(10)
    tree.fit(X, Y)
    pred = tree.predict(X)
    print(pred)


if __name__ == '__main__':
    main()
