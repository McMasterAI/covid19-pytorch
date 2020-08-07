import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wget
from sklearn.preprocessing import MinMaxScaler


def download_csv(url, save_location):
    """Downloads a csv file from the specified url to the specified save location.

    Args:
        url (str): url to download csv file.
        save_location (str): absolute path to save the downloaded file.
    """
    if os.path.exists(save_location):
        os.remove(save_location)
    wget.download(url, save_location)


def get_date_list(start="2020-01-23", end=datetime.now().strftime("%Y-%m-%d")):
    """Generates a list of dates in the format YYYY-MM-DD between the specified start and end dates.

    Args:
        start (str, optional): Start date in the format 'YYYY-MM-DD'. Defaults to '2020-01-23'.
        end (str, optional): End date in the format 'YYYY-MM-DD'. Defaults to the current day.

    Returns:
        [str]: Numpy array of dates between the start and end date. Formatted as 'YYYY-MM-DD'.
    """
    return pd.date_range(start=start, end=end).strftime("%Y-%m-%d").to_numpy()


def process_csv(save_location):
    """Retrives a list of dates and confimed cases from a downloaded csv file.

    Args:
        save_location (str): Absolute path of the csv file location.

    Returns:
        [str], [int]: A list of unique dates representing the dates where there were confirmed cases, the number of confirmed cases for each given date.
    """
    cr = csv.reader(open(save_location, "r"))
    next(cr)
    formatted = []
    for row in cr:
        if row[2] != "":
            formatted.append(row[2])
    np_array = np.array(formatted)
    unique_elements, counts_elements = np.unique(np_array, return_counts=True)
    return unique_elements, counts_elements


def interpolate_cases(unique, counts):
    """Interpolates number of confirmed cases for dates that did not have a recorded number of confirmed cases.

    Args:
        unique ([str]): List of dates having number of confirmed COVID-19 cases.
        counts ([int]): List of corresponding counts of cases for the given date.

    Returns:
        [[str], [int]]: Complete list of dates with corresponding case numbers.
    """
    full_date_list = get_date_list(end=str(unique[-1]))
    complete_date_array = [[], []]

    for date in full_date_list:
        try:
            complete_date_array[0].append(date)
            complete_date_array[1].append(int(counts[np.where(unique == date)]))
        except:
            complete_date_array[1].append(np.nan)
            continue

    s = pd.Series(complete_date_array[1])
    complete_date_array[1] = s.interpolate(limit_direction="backward").to_list()

    return complete_date_array


def normalize_data(to_normalize):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(to_normalize.reshape(-1, 1))
    return normalized_data


def create_tensors(normalized_train_data):
    train_window = 7  # 7 days in a week
    normalized_train_data = torch.FloatTensor(normalized_train_data).view(-1)

    inout_seq = []
    L = len(normalized_train_data)
    for i in range(L - train_window):
        train_seq = normalized_train_data[i : i + train_window]
        train_label = normalized_train_data[i + train_window : i + train_window + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def plot(unique_elements, counts_elements):
    plt.plot(unique_elements, counts_elements)
    plt.show()


def main():
    url = "https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv"
    save_location = os.getcwd() + "/temp/conposcovidloc.csv"

    test_size = 15

    download_csv(url, save_location)
    unique, counts = process_csv(save_location)
    data_array = interpolate_cases(unique, counts)
    plot(data_array[0], data_array[1])
    print()
    print(np.array(data_array[1][:-test_size]))
    normalized_train_data = normalize_data(np.array(data_array[1][:-test_size]))
    inout_seq = create_tensors(normalized_train_data)
    print(inout_seq)


if __name__ == "__main__":
    main()
