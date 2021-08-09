import numpy as np


class ConfusionMatrix(object):

    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0
        self.precision = 0
        self.sensitivity = 0
        self.specificity = 0
        self.f1 = 0

    def update(self, prediction, truth):
        self.tp += 1 if (truth == 1) and (prediction == 1) else 0
        self.tn += 1 if (truth == 0) and (prediction == 0) else 0
        self.fp += 1 if (truth == 0) and (prediction == 1) else 0
        self.fn += 1 if (truth == 1) and (prediction == 0) else 0

        if self.tp + self.fp + self.fn + self.tn > 0:
            self.accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

        if self.tp + self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)

        if self.tp + self.fn > 0:
            self.sensitivity = self.tp / (self.tp + self.fn)

        if self.tn + self.fp > 0:
            self.specificity = self.tn / (self.tn + self.fp)

        if self.precision + self.sensitivity > 0:
            self.f1 = 2 * self.precision * self.sensitivity / (self.precision + self.sensitivity)

    def get_metrics(self):
        return {
            'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn, 'accuracy': self.accuracy, 'precision': self.precision, 'sensitivity': self.sensitivity, 'specificity': self.specificity,
            'f1': self.f1
        }

    def print_matrix(self):
        cm = np.zeros((2, 2))
        cm[0, 0] = self.tp
        cm[0, 1] = self.fp
        cm[1, 0] = self.fn
        cm[1, 1] = self.tn

        print('\nConfusion_matrix:')
        for i in cm:
            print(i)
        print()

