import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, max_iter, fit_intercept):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.weights = []
    
    def sigmoid(self, s):
        return 1.0/(1 + np.exp(-s))
    
    def compute_prediction(self, X, weights):
        s = np.dot(X, weights)
        return self.sigmoid(s)

    def predict(self, X):
        predict = []
        
        if self.fit_intercept:
            X['intercept'] = np.ones((X.shape[0], 1))
        
        predict = np.round(self.compute_prediction(X, self.weights))
        return predict
    
    def compute_cost(self, X, y, weights):
        predictions = self.compute_prediction(X, weights)
        return np.mean(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    
    def update_weights(self, X_train, y_train, weights, learning_rate):
        for i in range (X_train.shape[0]):
            X = X_train.iloc[i]
            y = y_train.iloc[i]
            predictions = self.compute_prediction(X, weights)
            delta_weights = X.T * (y - predictions)
            weights += learning_rate * delta_weights
        return weights
    
    def train_logistic_regression(self, X_train, y_train):
        if self.fit_intercept:
            intercept = np.ones((X_train.shape[0], 1))
            X_train['intercept'] = intercept
        
        self.weights = np.zeros(X_train.shape[1])

        before_cost = 1
        for _ in range(self.max_iter):
            self.weights = self.update_weights(X_train, y_train, self.weights, self.learning_rate)
            now_cost = self.compute_cost(X_train, y_train, self.weights)
            # print(now_cost)
            if (before_cost-now_cost) < 1e-8:
                break
            before_cost = now_cost