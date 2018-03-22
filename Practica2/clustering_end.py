# -*- coding: utf-8 -*-
"""
Autor:
    Alberto Argente del Castillo Garrido
Fecha:
    Diciembre 2017
Contenido:
    Clustering en Python
    Inteligencia del Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

# SELECCIÓN DE CARACTERÍSTICAS: BORUTA ALGORITHM

import time
import random

random.seed(9654852698)
# For debugging
# import pdb

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans, Birch, DBSCAN, MeanShift
from sklearn.cluster import AgglomerativeClustering, estimate_bandwidth
from sklearn import metrics
from sklearn import preprocessing
from math import floor
import seaborn as sns

def getDataSet(X,variables, remove):
    X = X[variables]
    categories = [x for x in X.columns if (X[x].dtype == object)]

    X = pd.get_dummies(X, columns=categories)

    X_normalized = preprocessing.normalize(X, norm='l2')
    X_normalized = pd.DataFrame(X_normalized,index=X.index,columns=list(X.columns.values))
    return [X,X_normalized]

def cluster_means(X: pd.DataFrame, clustering_col: str = 'cluster'):
    """Create a dataframe with the mean of each cluster."""
    # Create an empty data frame
    X_means = pd.DataFrame()

    # Iterate every column in X
    for attr in X.columns:

        # If the column is not the clustering one
        if attr != clustering_col:

            # Perform the mean function to every value in the same cluster
            X_means[attr] = X.groupby(clustering_col)[attr].mean()
        X_means_normalized = preprocessing.normalize(X_means, norm='l2')
        X_means_norm = pd.DataFrame(X_means_normalized,index=X_means.index,columns=list(X_means.columns.values))
    return [X_means, X_means_norm]


t_global_start = time.time()

accidentes = pd.read_csv('accidentes_2013.csv')


# seleccionar accidentes de tipo 'colisión de vehículos' por filas
#subset = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]

# seleccionar accidentes en la autovía
subset = accidentes.loc[accidentes['TIPO_VIA'].str.contains("AUTOVÍA")]

#subset = accidentes[
#    (accidentes['TIPO_ACCIDENTE'].str.contains('Colisión de vehículos')) &
#    (accidentes['DIASEMANA'] <= 5) & (accidentes['TOT_VICTIMAS'] >= 0)
#]

#seleccionar solo accidentes entre las 6 y las 12 de la mañana por columnas
#subset1 = accidentes.loc[
#    (accidentes['TOT_VICTIMAS']!=0) &
#    (subset['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")) &
#    (accidentes['COMUNIDAD_AUTONOMA'].str.contains('Andalucía'))
#]
subset1 = subset.loc[
    # (accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")) &
    (accidentes['DIASEMANA'] < 5)
]
#seleccionar accidentes mortales
#subset1 = accidentes.loc[
#    (accidentes['TOT_VICTIMAS']!=0) &
    # (accidentes['TOT_HERIDOS_LEVES']!=0) &
    # (accidentes['TOT_HERIDOS_GRAVES']!=0)
    # (accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")) &
    # (accidentes['PROVINCIA'].str.contains("Barcelona")) &
    # (accidentes['TOT_MUERTOS']!=0)
    # (accidentes['TIPO_INTERSEC'].str.contains("NO_ES_INTERSECCION")) &
    # (accidentes['ZONA_AGRUPADA'].str.contains("VÍAS INTERURBANAS"))
#]

#seleccionar variables de interés para clustering
#usadas = ['HORA', 'DIASEMANA', 'TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

# Accidentes con una densidad de circulación densa
subset2 = accidentes.loc[
    (accidentes['DENSIDAD_CIRCULACION'].str.contains("DENSA")) &
    (accidentes['TIPO_INTERSEC'].str.contains("NO_ES_INTERSECCION"))
]


# Preparing the subsets

global_subsets = [subset1]

usadas = [
    'TOT_HERIDOS_LEVES', 'TOT_HERIDOS_GRAVES','TOT_VEHICULOS_IMPLICADOS'
]

#usadas = ['MES', 'TOT_MUERTOS', 'FACTORES_ATMOSFERICOS']

usadas1 = [
    'TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES',  'TOT_HERIDOS_LEVES','TOT_VEHICULOS_IMPLICADOS'
]
usadas2 = [
    'TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_VEHICULOS_IMPLICADOS'
]



global_usadas = [usadas1]
# %%
# Preparing the algorithm
print('Preparing algorithm')
"""
k_means = KMeans(init='k-means++', n_clusters=4, n_init=5)
birch = Birch(branching_factor=50, n_clusters=25, threshold=0.1,
            compute_labels=True)
dbs = DBSCAN(eps=0.1)
ms = MeanShift()
ward = AgglomerativeClustering(n_clusters=100, linkage="ward",
                             affinity="euclidean")

"""
"""
J = getDataSet(dataset, varis, None)
bandwidth = estimate_bandwidth(J[1], quantile=0.2, n_samples=500)
k_means = KMeans(init='k-means++', n_clusters=5, n_init=8)
birch = Birch(branching_factor=75, n_clusters=50, threshold=0.2,
            compute_labels=True)
dbs = DBSCAN(eps=0.1, min_samples = 5)
ms = MeanShift(bandwidth)
ward = AgglomerativeClustering(n_clusters=200, linkage="ward",
                             affinity="euclidean")
"""

J = getDataSet(dataset, varis, None)
bandwidth = estimate_bandwidth(J[1], quantile=0.2, n_samples=2500)
k_means = KMeans(init='k-means++', n_clusters=7, n_init=7)
birch = Birch(branching_factor=30, n_clusters=100, threshold=0.3,
            compute_labels=True)
dbs = DBSCAN(eps=0.2, min_samples = 20)
ms = MeanShift(bandwidth)
ward = AgglomerativeClustering(n_clusters=25, linkage="ward",
                            affinity="euclidean")

clustering_algorithm  = (
    ('K-means', k_means),
    ('Birch', birch),
    ('DBSCAN', dbs),
    ('Mean Shift', ms),
    ('A.G. Ward', ward)
)
# Dict where we will save all the results
global_results = dict()
# %%
for index, (dataset, varis) in enumerate(zip(global_subsets, global_usadas)):
    # Y = dataset[varis]
    Y = getDataSet(dataset, varis, None)
    # X = subset[dataset]
    # X = X.sample(1000)
    # X_normal = preprocessing.normalize(X, norm='l2')
    X = Y[0]
    # X.sample(1000)
    X_normal = Y[1]
    # Results for every execution
    cases_results = dict()
    print('Starting execution')
    # pdb.set_trace()
    for name, algorithm in clustering_algorithm:
        print('{:19s}'.format(name), end = ' ')
        # Dict where we will save the results
        current_results = dict()

        # Init of execution
        t = time.time()
        cluster_predict = algorithm.fit_predict(X_normal)
        tiempo = time.time() - t
        # End of execution

        # Number of predicted clusters
        k = len(set(cluster_predict))
        aux_k = k
        # Print clusters and time
        print(': k: '.format(k), end='')
        print('{:6.2f} segundos, '.format(tiempo), end = '' )

        # If it's not AgglomerativeClustering or Birch
        if (k>1) and ((name != 'A.G. Ward') and (name != 'Birch')):
            print('Calculating metrics.')
            # Calinsky metric
            metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
            # Silhouette metric
            metric_SC = metrics.silhouette_score(X_normal, cluster_predict,
                metric='euclidean', sample_size=floor(0.1*len(X)), random_state=123456)
            # Printing results
            print('CH Index: {:9.3f}, '.format(metric_CH), end = '')
            print('SC Index: {:.5f}'.format(metric_SC))

            current_results['k_clusters'] = k
            current_results['SC-metric'] = metric_SC
            current_results['CH-metric'] = metric_CH
            current_results['Time'] = tiempo
            current_results['Clusters_predict'] = cluster_predict

            cases_results[name] = current_results
            # Print matrix left
            # Adding cluster column
            clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
            X_alg = pd.concat([X, clusters], axis=1)
            print("---------- Preparing the scatter matrix...")
            sns.set()
            variables = list(X_alg)
            variables.remove('cluster')
            sns_plot = sns.pairplot(X_alg, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
            sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
            sns_plot.savefig(name + str(index) + "SM.png")
            print("")
        else:
            if (name == 'A.G. Ward'):
                # Quitting outliers from tiny clusters in hierarchy algorithm
                # se convierte la asignación de clusters a DataFrame
                clusters = pd.DataFrame(cluster_predict, index=X.index, columns=['cluster'])
                # y se añade como columna a X
                X_cluster = pd.concat([X, clusters], axis=1)
                min_size = 15
                X_filtered = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
                k_filtered = len(set(X_filtered['cluster']))
                print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k, k_filtered, min_size, len(X), len(X_filtered)))
                X_filtered = X_filtered.drop('cluster', 1)
                # Normalize the filtered set
                X_filtered_normal = preprocessing.normalize(X_filtered, norm='l2')
                # Prepare Ward with filtered data
                ward_filtered = AgglomerativeClustering(n_clusters=k_filtered, linkage="ward",
                                             affinity="euclidean")
                # Init of execution
                t = time.time()
                cluster_predict_filtered = ward_filtered.fit_predict(X_filtered_normal)
                tiempo_filtered = time.time() - t
                # End of execution
                """
                Hemos eliminado los outliers y luego volvemos a ejecutar el algoritmo
                de Ward pero esta segunda vez elegimos para ejecutar el AG
                con los clusters que hemos obtenido al filtrar (k_filtered)
                y luego ya no hace falta que volvamos a filtrar para poder
                pintar las gráficas correctamente.
                Una vez tenemos los resultados filtrados calculamos las métricas
                """
                print('Calculating metrics.')
                # Calinsky metric
                metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
                # Silhouette metric
                metric_SC = metrics.silhouette_score(X_normal, cluster_predict,
                    metric='euclidean', sample_size=floor(0.1*len(X)), random_state=123456)
                # Printing results
                print('CH Index: {:9.3f}, '.format(metric_CH), end = '')
                print('SC Index: {:.5f}'.format(metric_SC))
                # Saving results
                current_results['k_clusters'] = k_filtered
                current_results['SC-metric'] = metric_SC
                current_results['CH-metric'] = metric_CH
                current_results['Time'] = tiempo_filtered
                current_results['Clusters_predict'] = cluster_predict_filtered

                cases_results[name] = current_results
                # Add again cluster column
                clusters_filt = pd.DataFrame(cluster_predict_filtered, index=X_filtered.index, columns=['cluster'])
                # y se añade como columna a X
                X_cluster_filt = pd.concat([X_filtered, clusters_filt], axis=1)
                # First we calculate the means of the clusters
                X_mean = cluster_means(X_cluster_filt, 'cluster')
                X_filtered_mean = X_mean[0]
                X_filtered_mean_normal = X_mean[1]
                # Using Scipy we obtain the dendrograma
                from scipy.cluster import hierarchy
                linkage_array = hierarchy.ward(X_filtered_mean_normal)
                plt.figure(1)
                plt.clf()
                # We put it like this to compare it with seaborn plot
                #hierarchy.dendrogram(linkage_array, orientation='left')
                # Now we obtain dendrogram with the heatmap
                #print(X_filtered.columns)
                #print(X_filtered_normal.columns)
                # X_filtered_mean_normal_DF = pd.DataFrame(X_filtered_mean_normal, index = X_filtered_mean.index, columns = list(X_filtered_mean))
                aux_fig = sns.clustermap(X_filtered_mean, method = 'ward', col_cluster = True, annot = True,figsize = (20,10), cmap="YlGnBu", yticklabels=False)
                # print(index)
                aux_fig.savefig(name + str(index) + 'HM.png')
                plt.show()

            else:
                # Birch
                # se convierte la asignación de clusters a DataFrame
                clusters = pd.DataFrame(cluster_predict, index=X.index, columns=['cluster'])
                # y se añade como columna a X
                X_cluster = pd.concat([X, clusters], axis=1)
                # Quitting outliers from tiny clusters in hierarchy algorithm
                min_size = 7
                X_filtered = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
                k_filtered = len(set(X_filtered['cluster']))
                print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k, k_filtered, min_size, len(X), len(X_filtered)))
                X_filtered = X_filtered.drop('cluster', 1)
                # Normalize the filtered set
                X_filtered_normal = preprocessing.normalize(X_filtered, norm='l2')
                birch_filtered = Birch(branching_factor=50, n_clusters=k_filtered, threshold=0.1,
                            compute_labels=True)
                # Init of execution
                t = time.time()
                cluster_predict_filtered = birch_filtered.fit_predict(X_filtered_normal)
                tiempo_filtered = time.time() - t
                # End of execution
                print('Calculating metrics.')
                # Calinsky metric
                metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
                # Silhouette metric
                metric_SC = metrics.silhouette_score(X_normal, cluster_predict,
                    metric='euclidean', sample_size=floor(0.1*len(X)), random_state=123456)
                # Printing results
                print('CH Index: {:9.3f}, '.format(metric_CH), end = '')
                print('SC Index: {:.5f}'.format(metric_SC))
                # Saving results
                current_results['k_clusters'] = k_filtered
                current_results['SC-metric'] = metric_SC
                current_results['CH-metric'] = metric_CH
                current_results['Time'] = tiempo_filtered
                current_results['Clusters_predict'] = cluster_predict_filtered

                cases_results[name] = current_results

        """
        print('Printing graphics')
        # Convert the cluster to a DataFrame
        clusters = pd.DataFrame(cluster_predict, index = X.index, columns=['cluster'])
        # Then add to column at X
        X_cluster = pd.concat([X,clusters], axis=1)
        print('Preparing Scatter Matrix')
        sns.set()
        variables = list(X_cluster)
        variables.remove('cluster')
        sns_plot = sns.pairplot(X_cluster, vars = variables, hue = "cluster", palette = 'Paired', plot_kws={"s": 25}, diag_kind="hist")
        sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
        print(name + 'png')
        sns_plot.savefig(name + 'SC.png')
        print("")
        # For AGWard
        # Quitting outliers from tiny clusters in the hierarchy graph
        min_size = 3
        X_filtered = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
        k_filtered = len(set(X_filtered['cluster']))
        print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k, k_filtered, min_size, len(X), len(X_filtered)))
        X_filtered = X_filtered.drop('cluster', 1)


        # Normalize the filtered set
        X_filtered_normal = preprocessing.normalize(X_filtered, norm='l2')

        # Using Scipy we obtain the dendrograma
        from scipy.cluster import hierarchy
        linkage_array = hierarchy.ward(X_filtered_normal)
        plt.figure(1)
        plt.clf()
        # We put it like this to compare it with seaborn plot
        hierarchy.dendrogram(linkage_array, orientation='left')

        # Now we obtain it using seaborn (and we include a heatmap)
        sns.set()
        X_filtered_normal_DF = pd.DataFrame(X_filtered_normal, index = X_filtered.index, columns = varis)
        sns.clustermap(X_filtered_normal_DF, method = name, col_cluster = False, figsize = (20,10), cmap="YlGnBu", yticklabels=False)
        sns.savefig(name + 'HM.png')
        plt.show()
        """

    global_results[index] = cases_results
# %%
for index, case in enumerate(global_subsets):
    print('\nStudy case {0}: '.format(index))
    print('\n{0:<15}\t{1:<10}\t{2:<10}\t{3:<10}\t{4:<10}'.format(
        'Name', 'N clusters', 'SC metric', 'CH metric', 'Time (s)'))

    for name, res in global_results[index].items():
        print('{0:<15}\t{1:<10}\t{2:<10.2f}\t{3:<10.2f}\t{4:<10.2f}'.format(
            name, res['k_clusters'], res['SC-metric'], res['CH-metric'],
            res['Time']))

    # print('\nTotal time = {0:.2f}'.format(time.time() - t_global_start))


    """
    Podemos cortar cuando el numero de clusters sea menor de 10 por ejemplo, que no
    van a servir de mucho para los gerarquicos para crear dendrograma.

    pintar: hierarchy denrogram heatmap
    https://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
    https://plot.ly/python/dendrogram/

    para pintar kmeans - scatterplot
    para pintar gerarquicos - dendrograma
    """
    """
    import seaborn as sns
    """
    #print("-------------- Preparando el scatter matrix...")
    # Se convierte la asignación de clusters a DataFrame
    #clusters = pd.DataFrame(cluster_predict['Ward'], index=X.index,columns=['cluster'])
    # Y se añadaa la columna X
    #X_clust = pd.concat([X,clusters],axis=1)
    #hierar = X_clust['cluster'].value_counts() # Esta línea y la siguiente son para
    #hierar = hierar[hierar>5]   # Estudiar los clusters
    #xxx = X_clust[X_clust.gropby('cluster').cluster.transform(len) > 5]
