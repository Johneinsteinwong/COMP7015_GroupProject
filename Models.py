import numpy as np
from scipy.stats import norm, logistic
from scipy.optimize import minimize
from sklearn.exceptions import NotFittedError

class LatentVariableModel():
    
    def __init__(self, dist, l1, l2):
        if l1<0 or l2<0:
            raise ValueError('l1 and l2 must be non-negative!')
        self.param = None
        self.dist = dist
        self.l1 = l1
        self.l2 = l2

    def __loss(self, param, x, y):
        p = param[:-1]
        b = param[-1]
        score = np.dot(x,p) + b
        diff = -y*self.dist.logcdf(score, loc=0, scale=1) - (1-y)*self.dist.logcdf(-score, loc=0, scale=1)
        obj = np.mean(diff) + self.l1*np.linalg.norm(param, ord=1) + self.l2*np.linalg.norm(param, ord=2)
        return obj

    def fit(self, x, y):
        loss_values = []
        self.param = np.zeros(x.shape[1]+1)
        def callback(param):
            current_loss = self.__loss(param, x, y)
            loss_values.append(current_loss)
            print("Current loss:", current_loss)
        
        result = minimize(self.__loss, self.param, args=(x, y), method='BFGS', tol=1e-3, options={'maxiter':1000}, callback=callback)
        self.param = result.x
        return result, loss_values
    
    def predict(self, x):
        if self.param is None:
            raise NotFittedError("Model is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        p = self.param[:-1]
        b = self.param[-1]
        score = np.dot(x,p) + b
        return score
    
    def predict_proba(self, x):
        score = self.predict(x)
        return self.dist.cdf(score, loc=0, scale=1)
    
class logisticModel(LatentVariableModel):
    def __init__(self, l1=0.0, l2=0.0):
        super().__init__(logistic, l1, l2)

class probitModel(LatentVariableModel):
    def __init__(self, l1=0.0, l2=0.0):
        super().__init__(norm, l1, l2)
