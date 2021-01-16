# Authors: Paweł Zborowski, Jakub Wirkus
# Problem: Train 2 models on 2 classification data sets

from LAB5.AdultExample import AdultExample
from LAB5.PimaIndiansExample import PimaIndiansExample


if __name__ == '__main__':
    print("------------------------Prima Indians example------------------------")
    primaIndians = PimaIndiansExample()
    primaIndians.run()

    print("----------------------------Adult example----------------------------")
    adult = AdultExample()
    adult.run()
