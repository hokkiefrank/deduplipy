from statistics import mode
def most_common(List):
    return (mode(List))


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def mapping(gtc, data, fc, cn):
    # dictionary where the key is the groundtruth clusternumber and the value the mapped found clusternumber
    returner = {}
    for i, ground_cluster in gtc.items():
        found_in = []
        results = {}
        for val in ground_cluster:
            clust_id = data.iloc[val][cn]
            if clust_id not in found_in:
                # intersection between one of the foundclusters with the groundtruth cluster
                result = intersection(ground_cluster, data.index[data[cn] == clust_id].tolist())
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


def evaluate(ground_truth_cluster, all_data, found_clusters, node_amount=10, coll_name="deduplication_id"):
    """
    Example usage: evaluate(groundtruth, res, res.groupby([markov_col]).indices, amount, markov_col)
    Args:
        ground_truth_cluster:
        all_data:
        found_clusters:
        node_amount:
        coll_name:

    Returns:

    """
    mapp = mapping(ground_truth_cluster, all_data, found_clusters, coll_name)
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



#print("Markov own:")
#evaluate_(groundtruth, res, res.groupby([markov_col]).indices, amount, markov_col)
#print("Hierar own:")
#evaluate_(groundtruth, res, res.groupby([hierar_col]).indices, amount, hierar_col)