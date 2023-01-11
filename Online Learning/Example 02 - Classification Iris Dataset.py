# IMPORTING LIBRARIES
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.datasets
import vowpalwabbit


class Dataframe_VW():

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


    def convert_dataframe_to_vw_format(self,
                                       dataframe,
                                       column_target="y"):

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
            self.__convert_row_to_vw_format(row=row,
                                            column_target=column_target)
            for idx, row in dataframe.iterrows()
        ]

        return data_vw_format


# IMPORTING DATA
iris_dataset = sklearn.datasets.load_iris()

# TRANSFORM DATA INTO DATAFRAME
iris_dataframe = pd.DataFrame(
    data=iris_dataset.data, columns=iris_dataset.feature_names
)

column_target = "y"

# vw expects labels starting from 1
iris_dataframe[column_target] = iris_dataset.target + 1

# CONVERTING DATAFRAME TO VW FORMAT
data_vw_format = Dataframe_VW().convert_dataframe_to_vw_format(
    dataframe=iris_dataframe, column_target=column_target
)

print(data_vw_format)
