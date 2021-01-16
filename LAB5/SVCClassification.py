import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC


class SVCClassification:
    """ Creates a model of given dataset """

    def __init__(self, dataset_file):
        self.dataset = pd.read_csv(dataset_file, sep=",", header=None)

    def set_columns(self, columns):
        """
        Sets column name for dataset
        :param columns: list of column names
        """
        self.dataset.columns = columns

    def get_model(self, output_column):
        """
        Calculates model and validates result.
        :param output_column: name of output data column
        :returns: SVC object
        """
        X = self.dataset.drop(output_column, axis='columns')
        y = self.dataset[output_column]
        print("Input data:")
        print(X)
        print("Output data")
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = SVC()
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_test, y_test, cv=10)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        return model





