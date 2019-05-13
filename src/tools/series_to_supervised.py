from pandas import DataFrame, concat


def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset
    Arguments:
        data: Sequence of observations as a list or numpy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (Y).
        dropnan: Boolean whether or not to drop rows with nan values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t - n, ..., t - 1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (columns[j - n_vars], i)) for j in range(n_vars)]
    # forecast sequence (t, t + 1, ..., t + n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % columns[j]) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (columns[j], i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with nan values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
