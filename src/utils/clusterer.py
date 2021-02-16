from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import numpy as np


class Clusterer:
    def __init__(self, text_to_vector_func, dataset, dist_threshold, dates=('2020-05-12', )):
        self.text_to_vector_func = text_to_vector_func
        self.dataset = dataset
        self.dist_threshold = dist_threshold
        self.dates = dates
        self.graph = np.array([list() for _ in range(len(self.dataset))])

    def perform_clustering(self):
        date_to_indices = self.__split_dataset_by_dates()

        for date in self.dates:
            embeds = np.empty((len(date_to_indices[date]), 768))
            subindex_to_index = dict()

            for i in range(embeds.shape[0]):
                dataset_index = date_to_indices[date][i]
                subindex_to_index[i] = dataset_index
                text = self.dataset[dataset_index]['title'] + ' ' + self.dataset[dataset_index]['text']
                text = text.lower().replace('\xa0', ' ').strip()
                embeds[i] = text_to_vector_func(text).detach().numpy().ravel()

            # TODO: precomputed affinity & custom distances
            clustering_model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.dist_threshold,
                linkage='single',
                affinity='cosine'
            )

            labels = clustering_model.fit_predict(embeds)

            for i1 in range(embeds.shape[0]):
                for i2 in range(i1 + 1, embeds.shape[0]):
                    if labels[i1] != labels[i2]:
                        continue

                    g1 = subindex_to_index[i1]
                    g2 = subindex_to_index[i2]
                    self.graph[g1].append(g2)
                    self.graph[g2].append(g1)

    def get_cluster_records(self, i):
        yield self.dataset.get_strings(i)

        for nei in self.graph[i]:
            yield self.dataset.get_strings(nei)

    def __split_dataset_by_dates(self):
        date_to_indices = defaultdict(list)

        for i, r in enumerate(self.dataset):
            cur_date = r['date'].split()[0]
            if cur_date in self.dates:
                date_to_indices[cur_date].append(i)

        return date_to_indices
