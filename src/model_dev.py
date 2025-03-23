import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

#abstract class for defining all models
class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
class LinearRegressionModel(Model):
    
    def train(self, X_train, y_train, **kwargs):
        try:
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Linear Regression model trained successfully")
            return model
        except Exception as e:
            logging.error("Error in training Linear Regression model: {}".format(e))
            raise e
    
    
    