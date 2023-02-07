import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def visualize_components_by_algorithm(population_data: dict, exclude: list[str], remove_outliers=False):
    """

    Args:
        remove_outliers: A boolean telling whether outliers should be removed from the visualisation
        population_data: A dictionary with the name of the method as key and a dataframe
        exclude: a list of strings with the methods names that should be excluded from the visualisation
    Returns:
        the plots
    """

    # get the first_dict to get the column names from
    first_dict = list(population_data.keys())[0]
    for (columnname, columndata) in first_dict.iteritems():
        temp_df = pd.DataFrame(columns=['variable', 'value'])
        for key in population_data.keys():
            if key in exclude:
                continue
            if columnname not in population_data[key]:
                continue
            temp = pd.DataFrame(population_data[key][columnname]).rename(
                columns={columnname: columnname + f"_{key}"}).melt()
            temp_df = pd.concat([temp_df, temp])
        if remove_outliers:
            temp_df = temp_df[~is_outlier(temp_df['value'])]
        plot = sns.histplot(temp_df, x='value', hue='variable', multiple='dodge', shrink=.75, bins=20)
        plt.show()
