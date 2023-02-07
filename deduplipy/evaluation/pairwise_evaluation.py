from entity_resolution_evaluation.evaluation import evaluate


def perform_evaluation(resulting_clusters: dict, ground_truth: list, verbose=True, evaluations="All") -> dict:
    """
    Performs evaluation from the paper by Menestrina et al.(2010)
    Args:
        resulting_clusters: A dict, where each key is a string indicating the name of
        the method used, to get the list of clusters.
        ground_truth: A list with the ground truth clusters
        verbose: whether to print the messages to the console
        evaluations: a list of strings with evaluations to use, standard is "All".
        It is possible to pick from ['precision', 'recall', 'f1', 'bmd', 'variation_of_information']

    Returns:
        A dictionary where firstly the key is the method name used for the cluster, with a dictionary as value.
        This dictionary has each evaluation method as key with the resulting value of that method as value
    """
    result = {}
    if evaluations == "All":
        evaluations = ['precision', 'recall', 'f1', 'bmd', 'variation_of_information']
    for r in resulting_clusters.keys():
        if verbose:
            print(f"------------------\nClustering method:{r}")
        result[r] = {}
        for eva in evaluations:
            result[r][eva] = evaluate(resulting_clusters[r], ground_truth, eva)
            if verbose:
                print(f"{eva}: {result[r][eva]:.4f}")

    return result
