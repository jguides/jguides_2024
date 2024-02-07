import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.jguides_2024.utils.df_helpers import zscore_df_columns
from src.jguides_2024.utils.plot_helpers import format_ax


class PCAContainer:

    def __init__(self, all_features_df, zscore_features=False, input_information=None):
        """
        Container for PCA input data, pca object, and results
        :param all_features_df: pandas df with features in columns and samples in rows
        :param zscore_features: True to zscore features
        :param input_information: additional information user wants to store
        """
        self.all_features_df = all_features_df
        self.zscore_features = zscore_features
        self.input_information = input_information
        self.input_df = self._get_input_df()
        self.input_array = self.input_df.to_numpy()
        self.pca_obj, self.pca_output_df, self.pca_output_arr = self._run_pca()

    def _get_input_df(self):
        if self.zscore_features:  # zscore each feature
            return zscore_df_columns(self.all_features_df)
        else:
            return self.all_features_df

    @staticmethod
    def format_pc_num(x):
        return f"pc_{x}"

    def _run_pca(self):
        pca = PCA(n_components=np.min(
            self.input_array.shape))  # n_components has to be less than or equal to the minimum of the dimension sizes
        pca.fit(self.input_array)
        pca_output_arr = pca.transform(self.input_array)
        pca_output_df = pd.DataFrame(pca_output_arr,
                   index=self.input_df.index,
                   columns=[self.format_pc_num(x) for x in np.arange(0, np.shape(pca_output_arr)[1])])
        return pca, pca_output_df, pca_output_arr

    def get_first_component_above_explained_variance_ratio(self, explained_variance_ratio_threshold):
        return np.where(np.cumsum(self.pca_obj.explained_variance_ratio_) > explained_variance_ratio_threshold)[0][0]

    def get_cumulative_explained_variance_ratio(self):
        return np.cumsum(self.pca_obj.explained_variance_ratio_)

    def plot_explained_variance_ratio(self, axes=None, explained_variance_ratio_thresholds=None):
        # Plot explained variance ratio and cumulative explained variance ratio
        # Check inputs
        if axes is not None:
            if len(axes) != 2:
                raise Exception(f"axes must be 2 dimensional")
        # Initialize plot if not passed
        if axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        # Variance explained ratio
        explained_variance_ratio = self.pca_obj.explained_variance_ratio_
        ax = axes[0]
        ax.plot(explained_variance_ratio, marker='o', color='k')
        format_ax(ax=ax, xlabel="principal component", ylabel="variance\nexplained ratio")
        # Cumulative variance explained ratio
        ax = axes[1]
        ax.plot(np.arange(0, len(explained_variance_ratio)), np.cumsum(explained_variance_ratio),
                marker='o', color='k')
        format_ax(ax=ax, xlabel="principal component", ylabel="cumulative variance\nexplained ratio")
        ax.set_yticks(np.arange(0, 1.1, step=0.2))

        # Plot explained_variance_ratio_thresholds if passed
        if explained_variance_ratio_thresholds is not None:
            for threshold in explained_variance_ratio_thresholds:
                component_num = self.get_first_component_above_explained_variance_ratio(threshold)
                explained_variance_ratio = self.get_cumulative_explained_variance_ratio()[component_num]
                ax.plot([component_num]*2, [0, explained_variance_ratio], color="red")
                ax.plot([0, component_num], [explained_variance_ratio]*2, color="red")

    def plot_pca_output_2D(
            self, plot_pc_1=0, plot_pc_2=1, fig_ax_list=None, alpha=.7, color="black"):
        if fig_ax_list is None:
            fig_ax_list = plt.subplots(figsize=(10, 10))
        fig, ax = fig_ax_list
        # Plot PCs
        ax.plot(self.pca_output_arr[:, plot_pc_1],
                self.pca_output_arr[:, plot_pc_2], marker='.', alpha=alpha, color=color)
        format_ax(ax=ax, xlabel=f'PC {plot_pc_1}', ylabel=f'PC {plot_pc_2}')
        # Remove axes
        _ = ax.set_xticks([])
        _ = ax.set_yticks([])
        for x in ["top", "bottom", "left", "right"]:
            ax.spines[x].set_visible(False)


def reconstruct_from_pcs(loadings, eigenvectors):
    """
    :param loadings: pca loadings. Shape: [number of eigenvectors, 1].
    :param eigenvectors: eigenvectors Shape: [number of features, number of eigenvectors].
    :return eigenvectors x loadings. Shape: [number of features, 1].
    """
    return eigenvectors @ loadings