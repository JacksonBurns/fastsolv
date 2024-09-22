from sklearn.decomposition import PCA
import umap
import pandas as pd

from chemplot import Plotter
import chemplot.parameters as parameters


class MoleculesUMAP(Plotter):
    def umap(self, n_neighbors=None, min_dist=None, pca=False, random_state=None, fit_on_target=None, **kwargs):
        """
        Provide fit_on_target to fit the projection to only a subset of the data
        """
        self.__data = self._Plotter__data_scaler()
        if fit_on_target is not None:
            fit_indices = [i for i in range(self.__data.shape[0]) if self._Plotter__target[i] == fit_on_target]
        else:
            fit_indices = list(range(self.__data.shape[0]))
        self.__plot_title = "UMAP plot"

        # Preprocess the data with PCA
        if pca and self._Plotter__sim_type == "structural":
            _n_components = 10 if len(self.__data[0]) >= 10 else len(self.__data[0])
            pca = PCA(n_components=_n_components, random_state=random_state)
            pca.fit(self.__data[fit_indices])
            self.__data = pca.transform(self.__data)
            self.__plot_title += " from components with cumulative variance explained " + "{:.0%}".format(sum(pca.explained_variance_ratio_))

        if n_neighbors is None:
            if self._Plotter__sim_type == "structural":
                if pca:
                    n_neighbors = parameters.n_neighbors_structural_pca(len(self.__data))
                else:
                    n_neighbors = parameters.n_neighbors_structural(len(self.__data))
            else:
                n_neighbors = parameters.n_neighbors_tailored(len(self.__data))

        # Get the perplexity of the model
        if min_dist is None or min_dist < 0.0 or min_dist > 0.99:
            if min_dist is not None and (min_dist < 0.0 or min_dist > 0.99):
                print("min_dist must range from 0.0 up to 0.99. Default used.")
            if self._Plotter__sim_type == "structural":
                if pca:
                    min_dist = parameters.MIN_DIST_STRUCTURAL_PCA
                else:
                    min_dist = parameters.MIN_DIST_STRUCTURAL
            else:
                min_dist = parameters.MIN_DIST_TAILORED

        # Embed the data in two dimensions
        self.umap_fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, n_components=2, **kwargs)
        self.umap_fit.fit(self.__data[fit_indices])
        ecfp_umap_embedding = self.umap_fit.transform(self.__data)
        # Create a dataframe containinting the first 2 UMAP components of ECFP
        self._Plotter__df_2_components = pd.DataFrame(data=ecfp_umap_embedding, columns=["UMAP-1", "UMAP-2"])

        if len(self._Plotter__target) > 0:
            self._Plotter__df_2_components["target"] = self._Plotter__target

        return self._Plotter__df_2_components.copy()
