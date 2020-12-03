from concurrent import futures
import numbers

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.series import Series
from sklearn.preprocessing import StandardScaler


def _column_to_ints(column: Series):
    """
    Converts a Pandas Series (aka column) from a list of strings to a list of
    unique indices. Each time a unique value is encountered, the index is
    incremented.
    """

    column_vals = {}
    for i in range(column.size):
        elem = column[i]
        if elem in column_vals:
            column[i] = column_vals[elem]
        else:
            new_val = len(column_vals)
            column[i] = new_val
            column_vals[elem] = new_val


def _handle_column(column):
    """
    Converts a column to integers (if required), then scales it
    """

    label: str = column[0]
    series: Series = column[1]
    series = series.copy()

    if not isinstance(series[0], numbers.Number):
        _column_to_ints(series)
    unscaled = series.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    # print(scaled)
    series.update(scaled)

    return label, series, unscaled

def _thresholds(data: DataFrame):
    newLabels = []
    for i in data.values:
        if i < -15: newLabels.append("< -15")
        elif -15 <= i <= -1: newLabels.append("[-15, -1]")
        elif i == 0: newLabels.append("0")
        elif i == 1: newLabels.append("1")
        elif i == 2: newLabels.append("2")
        elif i == 3: newLabels.append("3")
        elif i == 4: newLabels.append("4")
        elif i == 5: newLabels.append("5")
        elif 6 <= i <= 7: newLabels.append("[6, 7]")
        elif 8 <= i <= 10: newLabels.append("[8, 10]")
        elif 11 <= i <= 20: newLabels.append("[11, 20]")
        else: newLabels.append("> 20")
    return DataFrame(newLabels, columns=["Yards"])

def main(file_name):
    print("Reading...")
    df = pd.read_csv(file_name, low_memory=False, keep_default_na=False)
    print("Done Reading...")
    df = df[df['NflId'] == df['NflIdRusher']]
    df = df.reset_index(drop=True)
    labels = df['Yards'].to_frame()
    bins = _thresholds(labels)
    bins.to_csv('./data/bins.csv')
    labels.to_csv('./data/labels.csv')
    df = df.drop(columns=['GameId', 'PlayId', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'DisplayName',
                          'JerseyNumber', 'NflIdRusher', 'TimeHandoff', 'TimeSnap', 'Yards'])
    df.to_csv("./data/column_filtered.csv", index=False)
    cleaned = DataFrame()
    with futures.ProcessPoolExecutor() as executor:
        for _, result in zip(range(len(df.columns)), executor.map(_handle_column, df.iteritems())):
            print(f'{round(_ / len(df.columns) * 100, 2)}%')
            label, series, unscaled = result
            df[label] = series
            cleaned[label] = unscaled
    print("Writing...")
    df.to_csv('./data/scaled.csv', index=False)
    cleaned.to_csv('./data/cleaned.csv', index=False)
    print("Done Writing...")


if __name__ == '__main__':
    FILE_NAME = './data/train.csv'
    main(FILE_NAME)
