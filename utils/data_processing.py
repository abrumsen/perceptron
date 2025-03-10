import pandas as pd
import os
import random

def store_data(dataframe:pd.DataFrame, file_path:str):
    """
    Creates or completes a csv based on the supplied dataframe
    :param dataframe: Pandas dataframe containing the various data generated by the perceptron during training.
    :param file_path: Path to the csv file containing the data to be stored.
    :return: returns None
    """
    if os.path.isfile(file_path):
        df = load_dataframe_from_file(file_path)
        frames = [dataframe, df]
        big_dataframe = pd.concat(frames, axis=0)
    else:
        big_dataframe = dataframe
    big_dataframe = big_dataframe.sort_values(by="Iteration")
    big_dataframe.to_csv(file_path, index=True, sep=";", mode="w")

def p_data_to_dataframe(iteration:int, weights:list, variables:list, obtained_value:float, expected_value:float):
    """
    Retrieves the various data generated by the perceptron and stores them in a pandas dataframe.
    :param iteration: Iteration number of the loop
    :param weights: Weights of the perceptron during the iteration.
    :param variables: Variables of the perceptron during the iteration.
    :param obtained_value: Obtained value of the perceptron during the iteration.
    :param expected_value: Expected value of the perceptron during the training.
    :return: returns a Pandas Dataframe
    """
    p_data = {
        "Iteration" : [iteration],
        "Weights":[weights],
        "Variables":[variables],
        "Obtained_value": obtained_value,
        "Expected_value": expected_value
    }
    return pd.DataFrame(p_data)

def load_dataframe_from_file(file_name:str="data.csv"):
    """
    This function loads the data from the csv file and returns it as a Pandas Dataframe.
    :param file_name: Name of the file containing the data to be loaded.
    :return: returns a Pandas Dataframe
    """
    return pd.read_csv(file_name, index_col=0, sep=";")

def generate_random_data(file_path:str, iteration_number):
    """
    generates a dataframe containing random values to mimic the result of perceptron training
    :param file_path: Path to the csv file containing the data to be stored.
    :param iteration_number: Iteration number of the loop
    :return: returns None
    """
    file_path = file_path
    num_iterations = iteration_number
    for i in range(1, num_iterations + 1):
        weights = [round(random.uniform(-1, 1), 2) for _ in range(2)]
        variables = [round(random.uniform(0, 1), 2) for _ in range(2)]
        obtained_value = round(sum(w * v for w, v in zip(weights, variables)), 2)
        expected_value = round(obtained_value + random.uniform(-0.5, 0.5), 2)
        df = p_data_to_dataframe(i, weights, variables, obtained_value, expected_value)
        store_data(df, file_path)






