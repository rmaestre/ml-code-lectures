from typing import Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class Cervical:
    def generate(
        self,
    ) -> Tuple[np.array, np.array]:
        """
        Read from cervical cancer dataset
        """
        # Read dataset
        bank = pd.read_csv("datasets/cervical.csv", delimiter=",")
        # Fill NA with 0's
        bank = bank.fillna(0)
        # Split train and test
        train, test = train_test_split(bank, test_size=0.2)
        # Get batches
        X_train = train.iloc[:, :-1].to_numpy()
        y_train = train.iloc[:, -1].to_numpy()
        X_test = test.iloc[:, :-1].to_numpy()
        y_test = test.iloc[:, -1].to_numpy()
        return X_train, y_train, X_test, y_test
