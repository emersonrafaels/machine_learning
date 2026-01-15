# IMPORTING LIBRARIES
from pprint import pprint

import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.datasets
import vowpalwabbit
from sklearn.metrics import classification_report, accuracy_score

SEED = 42


class Dataframe_VW:

    def __init__(self):

        pass

    def __convert_row_to_vw_format(self, row, column_target="y"):
        """

        CONVERTINDO ROW (SERIES) TO VOWPAL WABBIT FORMAT

        EX:
            "0 | price:.23 sqft:.25 age:.05 2006"

        """

        # GETTING COLUMN TARGET
        res = f"{int(row[column_target])} |"

        for idx, value in row.drop([column_target]).items():

            feature_name = idx.replace(" ", "_").replace("(", "").replace(")", "")
            res += f" {feature_name}:{value}"

        return res

    def convert_dataframe_to_vw_format(self, dataframe, column_target="y"):
        """

        CONVERTINDO DATAFRAME TO VOWPAL WABBIT FORMAT

        EX:
            "0 | price:.23 sqft:.25 age:.05 2006"

        """

        # INIT LIST RESULT
        data_vw_format = []

        # INTERACTING DATAFRAM
        # USING LIST COMPREHESION
        data_vw_format = [
            self.__convert_row_to_vw_format(row=row, column_target=column_target)
            for idx, row in dataframe.iterrows()
        ]

        return data_vw_format


def orchestra_train_test_model(SEED=0):

    # INIT DICT RESULT
    result = {}

    # IMPORTING DATA
    iris_dataset = sklearn.datasets.load_iris()

    # TRANSFORM DATA INTO DATAFRAME
    iris_dataframe = pd.DataFrame(
        data=iris_dataset.data, columns=iris_dataset.feature_names
    )

    # COLUMN TARGET
    column_target = "y"

    # vw expects labels starting from 1
    iris_dataframe[column_target] = iris_dataset.target + 1

    # SPLITING TRAIN AND TEST DATASET
    training_data, testing_data = sklearn.model_selection.train_test_split(
        iris_dataframe, test_size=0.2, random_state=SEED
    )

    # CONVERTING DATAFRAME TO VW FORMAT - TRAIN
    data_vw_format_train = Dataframe_VW().convert_dataframe_to_vw_format(
        dataframe=training_data, column_target=column_target
    )

    # CONVERTING DATAFRAME TO VW FORMAT - TEST
    data_vw_format_test = Dataframe_VW().convert_dataframe_to_vw_format(
        dataframe=testing_data, column_target=column_target
    )

    # CREATING AN INSTANCE OF VOWPAL WABBIT
    # quiet=True, avoid diagnostic informations in console
    model = vowpalwabbit.Workspace("--oaa 3 --quiet")

    # LEARNING WITH TRAIN DATASET
    [model.learn(example) for example in data_vw_format_train]

    # DOING PREDICTIONS
    predictions = [model.predict(example) for example in data_vw_format_test]

    # GET METRICS
    result["support"] = len(predictions)
    result["accuracy"] = accuracy_score(
        y_pred=predictions, y_true=testing_data[column_target]
    )
    result["classification_report"] = classification_report(
        y_pred=predictions,
        y_true=testing_data[column_target],
        output_dict=True,
        zero_division=0,
    )

    return predictions, result


if __name__ == "__main__":

    predictions, results = orchestra_train_test_model(SEED=SEED)

    pprint(results)
