from concurrent import futures
import numbers

import pandas as pd
from pandas.core.series import Series
from sklearn.decomposition import PCA
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

    scaler = StandardScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    # print(scaled)
    series.update(scaled)

    return label, series

def _thresholds(data):
    data[data < 0] = 0
    data[(data >= 0) & (data < 2)] = 1
    data[(data >= 2) & (data < 4)] = 2
    data[(data >= 4) & (data < 6)] = 3
    data[(data >= 6) & (data < 8)] = 4
    data[(data >= 8) & (data < 10)] = 5
    data[(data >= 10) & (data < 15)] = 6
    data[(data >= 15) & (data < 30)] = 7
    data[data >= 30] = 8

def main(file_name):
    print("Reading...")
    df = pd.read_csv(file_name, low_memory=False, keep_default_na=False)
    print("Done Reading...")
    df = df[df['NflId'] == df['NflIdRusher']]
    df = df.reset_index(drop=True)
    labels = df['Yards'].to_frame()
    _thresholds(labels)
    labels.to_csv('./data/labels.csv')
    df = df.drop(columns=['GameId', 'PlayId', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'DisplayName',
                          'JerseyNumber', 'NflIdRusher', 'TimeHandoff', 'TimeSnap', 'Yards'])
    with futures.ProcessPoolExecutor() as executor:
        for _, result in zip(range(len(df.columns)), executor.map(_handle_column, df.iteritems())):
            print(f'{round(_ / len(df.columns) * 100, 2)}%')
            label, series = result
            df[label] = series
    print("Writing...")
    df.to_csv('./data/scaled.csv', index=False)
    print("Done Writing...")


if __name__ == '__main__':
    FILE_NAME = './data/train.csv'
    main(FILE_NAME)
