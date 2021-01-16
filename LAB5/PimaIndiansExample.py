# dataset source: https://machinelearningmastery.com/standard-machine-learning-datasets/
# more info: https://machinelearningmastery.com/standard-machine-learning-datasets/
from LAB5.SVCClassification import SVCClassification


class PimaIndiansExample:
    """ Creates a model of Pima Indians Diabetes dataset """

    def __init__(self):
        self.svc = SVCClassification('indian-dataset.txt')

    def run(self):
        """
        Runs example.
        :returns: SVC object
        """
        columns = ["No times pregnant", "Plasma glucose",
                   "blood pressure", "skinfold thickness", "2-Hour serum insulin",
                   "BMI", "Diabetes pedigree function", "Age", "Class"]
        self.svc.set_columns(columns)
        self.svc.get_model('Class')