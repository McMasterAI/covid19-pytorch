import os
from datetime import datetime, timedelta

import preprocess as pp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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

    epochs = 500

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

        if i % 25 == 1:
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


def main():
    url = "https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv"
    csv_location = os.getcwd() + "/temp/conposcovidloc.csv"
    model_location = os.getcwd() + "/temp/model.pt"

    # get data and create inout sequences
    download_new_file = False
    if download_new_file:
        pp.download_csv(url, csv_location)
    unique, counts = pp.process_csv(csv_location)
    data_array = pp.interpolate_cases(unique, counts)
    normalized_data = pp.normalize_data(np.array(data_array[1]))
    train_window = 30
    inout_seq = pp.create_tensors(normalized_data, train_window)

    # train model and make predictions
    num_forecast = 7
    train_new_model = False
    if train_new_model:
        model = train_model(inout_seq)
        torch.save(model, model_location)
    else:
        model = torch.load(model_location)
    normalized_preds = predict(model, num_forecast, normalized_data, train_window)
    predictions = pp.denormalize_data(normalized_preds)
    prediction_dates = pp.get_date_list(
        start=str(datetime.strptime(data_array[0][-1], "%Y-%m-%d") + timedelta(days=1)),
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


if __name__ == "__main__":
    main()