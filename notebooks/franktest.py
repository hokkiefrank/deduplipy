import statistics
from statistics import mode
from time import perf_counter
from entity_resolution_evaluation.evaluation import evaluate
from deduplipy.clustering.clustering import markov_clustering, hierarchical_clustering
from deduplipy.datasets import load_data
from deduplipy.deduplicator import Deduplicator
import pickle


def most_common(List):
    return (mode(List))


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def mapping(gtc, data, fc):
    returner = {}  # dictionary where the key is the groundtruth clusternumber and the value the mapped found clusternumber
    for i, ground_cluster in gtc.items():
        # high = 0
        # ind = -1
        # t1_start = perf_counter()
        found_in = []
        results = {}
        for val in ground_cluster:
            clust_id = data.iloc[val]["deduplication_id"]
            if clust_id not in found_in:
                # clust_size = len(data[data["deduplication_id"] == clust_id])
                result = intersection(ground_cluster, data.index[data[
                                                                     "deduplication_id"] == clust_id].tolist())  # intersection between one of the foundclusters with the groundtruth cluster
                results[clust_id] = len(result)  # / clust_size
            found_in.append(clust_id)
        f = max(results, key=results.get)
        returner[i] = f
    return returner


def precision(mapp, gtc, fc):
    pris = {}
    for i, ground_cluster in gtc.items():
        fgi = fc[mapp[i]]
        gi = ground_cluster
        up = len(intersection(fgi, gi))
        down = len(fgi)
        pri = up / down
        pris[i] = pri

    return pris


def recall(mapp, gtc, fc):
    pres = {}
    for i, ground_cluster in gtc.items():
        fgi = fc[mapp[i]]
        gi = ground_cluster
        up = len(intersection(fgi, gi))
        down = len(gi)
        pri = up / down
        pres[i] = pri

    return pres


def evaluate_(ground_truth_cluster, all_data, found_clusters, node_amount=10):
    mapp = mapping(ground_truth_cluster, all_data, found_clusters)
    precisions = precision(mapp, ground_truth_cluster, found_clusters)
    total_precision = 0
    for i, pre in precisions.items():
        total_precision += ((len(ground_truth_cluster[i]) / node_amount) * pre)

    recalls = recall(mapp, ground_truth_cluster, found_clusters)
    total_recall = 0
    for i, rec in recalls.items():
        total_recall += ((len(ground_truth_cluster[i]) / node_amount) * rec)

    f_measure = (2 * total_precision * total_recall) / (total_recall + total_precision)
    print(f"Total weighted precision: {total_precision}, Total weighted recall: {total_recall}\nF1: {f_measure}")


df = load_data(kind='musicbrainz20k')

amount = len(df)
group = df.groupby(['CID'])  # CID for musicbrainz, id for stoxx50
groundtruth = group.indices
myDedupliPy = Deduplicator(['title', 'artist', 'album'])  # ,voter: 'suburb', 'postcode'])
myDedupliPy.verbose = True
#myDedupliPy.fit(df)

#with open('musicbrainz20kfulltest.pkl', 'wb') as f:
#    pickle.dump(myDedupliPy, f)
with open('musicbrainz20kfulltest.pkl', 'rb') as f:
    myDedupliPy = pickle.load(f)

myDedupliPy.verbose = True
res = myDedupliPy.predict(df)
res_mc = myDedupliPy.predict(df, clustering=markov_clustering)
print(res.sort_values('deduplication_id').head(10))
print(res_mc.sort_values('deduplication_id').head(10))
sorted_res = res.sort_values('deduplication_id')
sorted_res_mc = res_mc.sort_values('deduplication_id')

start_eval_1 = perf_counter()
evaluate_(groundtruth, res, res.groupby(['deduplication_id']).indices, amount)
stop_eval_1 = perf_counter()
print(f'Evaluation 1 took:{stop_eval_1 - start_eval_1:.4f} seconds')

start_eval_2 = perf_counter()
evaluate_(groundtruth, res_mc, res_mc.groupby(['deduplication_id']).indices, amount)
stop_eval_2 = perf_counter()
print(f'Evaluation 2 took:{stop_eval_2 - start_eval_2:.4f} seconds')

r1 = list(res.groupby(['deduplication_id']).groups.values())
r2 = list(res_mc.groupby(['deduplication_id']).groups.values())
s = list(group.groups.values())
rs = [r1, r2]
evaluations = ['precision', 'recall', 'f1', 'bmd']
for r in rs:
    for eva in evaluations:
        print(f"{eva}: {evaluate(r,s,eva)}")

#print(evaluate(r, s, 'f1'))
#print(evaluate(r2, s, 'f1'))

print("done")
