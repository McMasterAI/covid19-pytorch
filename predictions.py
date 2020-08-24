import json
import os
from datetime import datetime, timedelta
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import preprocess as pp


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_model(inout_seq):
    """Trains the LSTM model.

    Args:
        inout_seq ([torch.Tensor]): In out sequence of tensors.

    Returns:
        object: LSTM model.
    """
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 150

    for i in range(epochs):
        for seq, labels in inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 0:
            print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

    print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

    return model


def predict(model, num_pred, train_data_normalized, train_window):
    """Makes predictions using LSTM model and normalized training data.

    Args:
        model (object): LSTM model.
        num_pred (int): Number of predictions to make.
        train_data_normalized (numpy.ndarray): Normalized data used to train the LSTM.
        train_window (int): Moving window used to aid in predictions.

    Returns:
        numpy.ndarray: Numpy array of predictions.
    """
    test_inputs = train_data_normalized[-train_window:]
    model.eval()
    for _ in range(num_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )
            test_inputs = np.append(test_inputs, model(seq).item())
    return test_inputs[-num_pred:]


def convert_json(save_location, locations_dict, prediction_dates=[]):
    new_dict = {}
    for location in locations_dict:
        current_loc = locations_dict[location]
        for datenum, date in enumerate(current_loc[0]):
            is_pred = True if date in prediction_dates else False
            if date in new_dict:
                new_dict[date].append(
                    {
                        "x": pp.coordinates[location]["x"],
                        "y": pp.coordinates[location]["y"],
                        "value": current_loc[1][datenum],
                        "loc": location,
                        "prediction": is_pred,
                    }
                )
            else:
                new_dict[date] = [
                    {
                        "x": pp.coordinates[location]["x"],
                        "y": pp.coordinates[location]["y"],
                        "value": current_loc[1][datenum],
                        "loc": location,
                        "prediction": is_pred,
                    }
                ]

    with open(save_location + "covidlocpreds.json", "w+") as f:
        return json.dump(new_dict, f)


def main():
    url = "https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv"
    csv_location = os.getcwd() + "/temp/conposcovidloc.csv"
    model_location = os.getcwd() + "/temp/model.pt"
    model_base_location = os.getcwd() + "/temp/"
    json_location = model_base_location

    # get data and create inout sequences
    download_new_file = True
    if download_new_file:
        pp.download_csv(url, csv_location)

    locations = True
    if not locations:
        unique, counts = pp.process_csv(csv_location)
        data_array = pp.interpolate_cases(unique, counts)
        scaler = pp.create_scaler()
        normalized_data = pp.normalize_data(np.array(data_array[1]), scaler)
        train_window = 7
        inout_seq = pp.create_tensors(normalized_data, train_window)

        # train model and make predictions
        num_forecast = 7
        train_new_model = True
        if train_new_model:
            model = train_model(inout_seq)
            torch.save(model, model_location)
        else:
            model = torch.load(model_location)
        normalized_preds = predict(model, num_forecast, normalized_data, train_window)
        predictions = pp.denormalize_data(normalized_preds, scaler)
        prediction_dates = pp.get_date_list(
            start=str(
                datetime.strptime(data_array[0][-1], "%Y-%m-%d") + timedelta(days=1)
            ),
            end=str(
                datetime.strptime(data_array[0][-1], "%Y-%m-%d")
                + timedelta(days=num_forecast)
            ),
        )

        # update case data with predictions
        data_array[0] = np.append(data_array[0], prediction_dates)
        data_array[1] = np.append(data_array[1], predictions)

        # plot original and predictions
        plt.title("Predictions")
        plt.ylabel("Confirmed Cases")
        plt.grid(True)
        plt.autoscale(axis="x", tight=True)
        plt.plot(data_array[0], data_array[1])
        plt.plot(prediction_dates, predictions)
        plt.show()
    else:
        # create dictionaries of data needed for each location
        locations_dict = pp.process_csv_locations(csv_location)
        date_list, _ = pp.process_csv(csv_location)
        interpolated_dict = {}
        scaler_dict = {}
        normalized_data = {}
        inout_locations = {}
        for location in locations_dict:
            # get unique element array and counts of elements array
            unique = locations_dict[location][0]
            counts = locations_dict[location][1]
            # pad with zeros for time series prediction
            interpolated_dict[location] = pp.interpolate_cases(
                unique, counts, zeros=True, end=str(date_list[-1])
            )

            # create a scaler for this location and normalize the data
            scaler_dict[location] = pp.create_scaler()
            normalized_data[location] = pp.normalize_data(
                np.array(interpolated_dict[location][1]), scaler_dict[location]
            )
            train_window = 7  # 1 week
            inout_locations[location] = pp.create_tensors(
                normalized_data[location], train_window
            )

            # train model and make predictions for each location
            num_forecast = 7
            train_new_model = False
            if train_new_model:
                model = train_model(inout_locations[location])
                torch.save(model, model_base_location + location + ".pt")
            else:
                model = torch.load(model_base_location + location + ".pt")

            # make predictions
            normalized_preds = predict(
                model, num_forecast, normalized_data[location], train_window
            )
            predictions = pp.denormalize_data(normalized_preds, scaler_dict[location])
            # create date list for predictions
            prediction_dates = pp.get_date_list(
                start=str(
                    datetime.strptime(interpolated_dict[location][0][-1], "%Y-%m-%d")
                    + timedelta(days=1)
                ),
                end=str(
                    datetime.strptime(interpolated_dict[location][0][-1], "%Y-%m-%d")
                    + timedelta(days=num_forecast)
                ),
            )

            # update case data with predictions
            interpolated_dict[location][0] = np.append(
                interpolated_dict[location][0], prediction_dates
            )
            interpolated_dict[location][1] = np.append(
                interpolated_dict[location][1], predictions
            )

            interpolated_dict[location][0] = interpolated_dict[location][0].tolist()
            interpolated_dict[location][1] = interpolated_dict[location][1].tolist()

        convert_json(json_location, interpolated_dict, prediction_dates)

        # plot actual cases and predictions
        plt.close()
        for location in interpolated_dict:
            plt.plot(
                interpolated_dict[location][0],
                interpolated_dict[location][1],
                label=location,
            )
        plt.title("COVID-19 Cases Per Location")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
