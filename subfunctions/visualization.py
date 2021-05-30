import matplotlib.pyplot as plt  # plotting
import matplotlib as mpl
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os  # accessing directory structure
import seaborn as sns


def plot_sensors_correlation(threshold_value, dataset):
    """Function plots the the correlation plots between sensor positions for each group"""
    list_of_pairs = []
    X = dataset.iloc[:, 1:]
    Y = dataset.iloc[:, 0]
    j = 0
    for column in X.columns:
        j += 1
        for i in range(j, len(X.columns)):
            if column != X.columns[i]:
                temp_pair = [column + '-' + str(X.columns[i])]
                list_of_pairs.append(temp_pair)

    df = X
    correlations = df.corr()
    fig = plt.figure(figsize=(17, 10))
    ax = fig.add_subplot(121)
    ax.set_title('User A', fontsize=14)
    mask = np.zeros_like(correlations, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(correlations, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.suptitle('Correlation between Sensor Positions for stimulus', fontsize=16)
    plt.show()

    get_correlated_pairs_sample(threshold=threshold_value, correlation_df=correlations, group='User A',
                                list_of_pairs=list_of_pairs)


def get_correlated_pairs_sample(threshold, correlation_df, group, list_of_pairs):
    # create dictionary wheke keys are the pairs and values are the amount of high correlation pair
    corr_pairs_dict = {}
    for i in range(len(list_of_pairs)):
        temp_corr_pair = dict(zip(list_of_pairs[i], [0]))
        corr_pairs_dict.update(temp_corr_pair)

    j = 0
    for column in correlation_df.columns:
        j += 1
        for i in range(j, len(correlation_df)):
            if ((correlation_df[column][i] >= threshold) & (column != correlation_df.index[i])):
                corr_pairs_dict[str(column) + '-' + str(correlation_df.index[i])] += 1

    corr_count = pd.DataFrame(corr_pairs_dict, index=['count']).T.reset_index(drop=False).rename(
        columns={'index': 'channel_pair'})
    print('Channel pairs that have correlation value >= ' + str(threshold) + ' (' + group + '):')
    print(corr_count['channel_pair'][corr_count['count'] > 0].tolist())


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[
        col] < 50]]  # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns')  # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# Scatter and density plots
def plotScatterMatrix(df, plot_size, text_size):
    df = df.select_dtypes(include=[np.number])  # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    column_names = list(df)
    if len(column_names) > 10:  # reduce the number of columns for matrix inversion of kernel density plots
        column_names = column_names[:10]
    df = df[column_names]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plot_size, plot_size], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center',
                          va='center', size=text_size)

    plt_name = 'Scatter and Density Plot'
    plt.suptitle(plt_name)
    plt.show()
    # save_plot(ax, plt_name)


def save_plot(ax, img_name):
    log_dir = 'content/dataVisual'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    img_path = os.path.join(log_dir, img_name)
    # plt.imsave(img_path, img)
    # ax.savefig(img_name)
    mpl.image.imsave(img_name, ax)


def plot_accuracy_loss(training_history):
    COLOR_AX1 = 'tab:red'
    COLOR_AX2 = 'tab:blue'
    COLOR_AX3 = 'tab:orange'
    COLOR_AX4 = 'tab:cyan'

    figure, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Cross Entropy Loss', color=COLOR_AX1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Training accuracy', color=COLOR_AX2)

    epochs = len(training_history.history['loss'])

    # ax1.plot(np.arange(0, epochs), training_history.history['loss'], color=COLOR_AX1)
    ax1.plot(np.arange(0, epochs), training_history.history['val_loss'], color=COLOR_AX3)
    # ax2.plot(np.arange(0, epochs), training_history.history['accuracy'], color=COLOR_AX2)
    ax2.plot(np.arange(0, epochs), training_history.history['val_accuracy'], color=COLOR_AX4)
    plt.show()
    plt.savefig('accuracy_loss_result.png')

