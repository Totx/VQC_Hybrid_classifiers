import pandas as pd

# Dataset reading, display and management of features and class variable.
def read_data(datadir, start_index, end_index, class_index, k_lines=10, separator=',',):
    dataf = pd.read_csv(datadir, sep=separator)
    with pd.option_context('display.max_rows', k_lines, 'display.max_columns', None):  # more options can be specified also
        print(dataf)
    xVals = dataf.iloc[:, start_index:end_index]
    yVals = dataf.iloc[:, class_index]
    return xVals, yVals


