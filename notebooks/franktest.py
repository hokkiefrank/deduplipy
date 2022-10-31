import statistics
from statistics import mode
from time import perf_counter
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
    returner = {} # dictionary where the key is the groundtruth clusternumber and the value the mapped found clusternumber
    for i, ground_cluster in gtc.items():
        #high = 0
        #ind = -1
        #t1_start = perf_counter()
        found_in = []
        results = {}
        for val in ground_cluster:
            clust_id = data.iloc[val]["deduplication_id"]
            if clust_id not in found_in:
                clust_size = len(data[data["deduplication_id"] == clust_id])
                result = intersection(ground_cluster, data.index[data["deduplication_id"] == clust_id].tolist()) #intersection between one of the foundclusters with the groundtruth cluster
                results[clust_id] = len(result)# / clust_size
            found_in.append(clust_id)
        f = max(results, key=results.get)
        #print(found_in)
        #print(f)
        #t1_stop = perf_counter()
        #print(f"Method 1 time:{t1_stop - t1_start}\n")
        #t2_start = perf_counter()
        #for j, cluster in fc.items():
        #    result = intersection(ground_cluster, cluster)
        #    if len(result) > high:
        #        high = len(result)
        #        ind = j
        #if ind == -1:
        #    print("something wrong")
        #t2_stop = perf_counter()
        #print(f"Method 2 time:{t2_stop - t2_start}\n")
        returner[i] = f
    #print(returner)
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


def evaluate(ground_truth_cluster, all_data, found_clusters, node_amount=10):
    mapp = mapping(ground_truth_cluster, all_data, found_clusters)
    precisions = precision(mapp, ground_truth_cluster, found_clusters)
    total_precision = 0
    for i, pre in precisions.items():
        total_precision += ((len(ground_truth_cluster[i])/node_amount) * pre)

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
myDedupliPy = Deduplicator(['title', 'artist']) #music: 'title', 'artist', 'album'])#,voter: 'suburb', 'postcode'])
myDedupliPy.verbose = True
myDedupliPy.fit(df)

with open('music20kartist.pkl', 'wb') as f:
    pickle.dump(myDedupliPy, f)
#with open('music20k.pkl', 'rb') as f:
#    myDedupliPy = pickle.load(f)

print(myDedupliPy)
myDedupliPy.verbose = True
res, res_mc = myDedupliPy.predict(df)

print(res.sort_values('deduplication_id').head(10))
sorted_res = res.sort_values('deduplication_id')
sorted_res_mc = res_mc.sort_values('deduplication_id')

start_eval_1 = perf_counter()
evaluate(groundtruth, res, res.groupby(['deduplication_id']).indices, amount)
stop_eval_1 = perf_counter()
print(f'Evaluation 1 took:{stop_eval_1 - start_eval_1} seconds')

start_eval_2 = perf_counter()
evaluate(groundtruth, res_mc, res_mc.groupby(['deduplication_id']).indices, amount)
stop_eval_2 = perf_counter()
print(f'Evaluation 2 took:{stop_eval_2 - start_eval_2} seconds')


print("done")
