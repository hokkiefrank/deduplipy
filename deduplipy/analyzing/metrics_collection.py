from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_completeness_v_measure

def perform_scoring(score, param, gt):

    if score == 'silhouette_score':
        return silhouette_score(param, gt, metric="precomputed")
    elif score == 'adjusted_rand_score':
        return adjusted_rand_score(gt, param)
    elif score == 'normalized_mutual_info_score':
        return normalized_mutual_info_score(gt, param)
    elif score == 'fowlkes_mallows_score':
        return fowlkes_mallows_score(gt, param)
    elif score == 'homogeneity_completeness_v_measure':
        return homogeneity_completeness_v_measure(gt, param)
    else:
        pass