# dataset source: https://archive.ics.uci.edu/ml/datasets/adult
from LAB5.SVCClassification import SVCClassification
import pandas as pd


class AdultExample:
    """ Creates a model of Adult dataset """

    def __init__(self):
        self.svc = SVCClassification('adult.data.txt')

    def run(self):
        """
        Runs example.
        :returns: SVC object
        """
        columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                   "marital-status", "occupation", "relationship", "race",
                   "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "earnings"]
        self.svc.set_columns(columns)
        self.__encode_categorical_data()
        return self.svc.get_model('earnings')

    def __encode_categorical_data(self):
        self.svc.dataset['earnings'].replace(['<=50K', '>50K'], [0, 1], inplace=True, regex=True)
        dummies = pd.get_dummies(self.svc.dataset).head(3500)
        self.svc.dataset = dummies
