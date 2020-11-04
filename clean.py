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

    return (label, series)

#'Team', 'PossessionTeam', 'FieldPosition', 'OffenseFormation', 'PlayDirection', 'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Stadium', 'StadiumType', 'Turf', 'GameWeather', 'WindDirection'

def main(file_name):
    print("Reading...")
    df = pd.read_csv(file_name, low_memory=False, keep_default_na=False)
    labels = df['Yards']
    labels.to_csv('labels.csv')
    df = df[df['NflId']==df['NflIdRusher']]
    df = df.drop(columns=['PlayId', 'GameId', 'DisplayName', 'JerseyNumber', 'Season', 'GameClock', 'OffensePersonnel', 'DefensePersonnel', 'TimeHandoff', 'TimeSnap', 'PlayerHeight', 'PlayerBirthDate', 'Location'], axis=1) # Work on keeping some of these in future
    with futures.ProcessPoolExecutor() as executor:
        for _, result in zip(range(len(df.columns)), executor.map(_handle_column, df.iteritems())):
            print(f'{round(_/len(df.columns)*100, 2)}%')
            label, series = result
            df[label] = series
    print("Writing...")
    df.to_csv('scaled.csv')
    print("Done Writing...")

if __name__ == '__main__':
    FILE_NAME = 'football.csv'
    main(FILE_NAME)
