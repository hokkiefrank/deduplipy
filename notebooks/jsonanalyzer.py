import itertools
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def filters(testrun, score_threshold):
    try:
        fullstring = testrun['changes_description']
        substring = "cora dataset" #  NEW RUNS: "cora_actual_dataset"  OLD RUNS: "cora dataset", "actual_settlemnts"
        if substring not in fullstring:
            #print(fullstring)
            return False

        substring2 = "weighted, unweighted"
        if substring2 not in fullstring:
            #print(fullstring)
            return False

        if testrun['config']['train_test_split'] != 0.2:
            #print(testrun['config']['train_test_split'])
            return False
        if testrun['config']['score_threshold'] != score_threshold:
            #print(testrun['config']['score_threshold'])
            return False
        print(f"THIS ONE IS INCLUDED: {fullstring}")
        return True
    except Exception as e:
        print(e)
        return False


def load_files(score):
    requesting = []
    directory = 'testruns'
    for filename in os.listdir(directory):
        name = os.path.join(directory, filename)
        print(name)
        try:
            ts = float(filename.replace(".json", ""))
            timeobj = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error: {e}")
            continue
        try:
            with open(name) as f:
                myDict = json.loads(f.read())
                myDict['timestamp'] = timeobj
                if filters(myDict, score):
                    requesting.append(myDict)
        except Exception as e:
            print(f"This file is bad: {name}")
    print(f"Found: {len(requesting)} files")
    return requesting


def get_average(testruns, actual_averages):
    algorithms = list(testruns[0]['label_dict'].keys())
    algorithms.remove('draw')
    if isinstance(testruns[0]['config']['ensemble_cut_probability'], list):
        for prob in testruns[0]['config']['ensemble_cut_probability']:
            algorithms.append('ensemble_no_weight_' + str(prob))
            algorithms.append('ensemble_weighted_' + str(prob))
    else:
        algorithms.append('ensemble_clustering')
    algorithms.append('mixed_best')
    for i in range(len(algorithms)):
        algo = algorithms[i]
        algorithms.append('test_split_'+algo)
    algorithms.append('test_split_predicted_clustering')
    averages = {}
    for run in testruns:
        for algo in algorithms:
            if algo not in averages:
                averages[algo] = {}
            if algo not in actual_averages:
                actual_averages[algo] = {}
            for key, value in run[algo].items():
                if key not in averages[algo]:
                    averages[algo][key] = []
                if key not in actual_averages[algo]:
                    actual_averages[algo][key] = []
                averages[algo][key].append(value)

    for algorithm, scores in averages.items():
        mes = f"{algorithm}:\t"
        for evaluation, values in scores.items():
            average = sum(values)/len(values)
            actual_averages[algorithm][evaluation].append(average)
            mes += f"{average:.4f}\t"
        print(mes)
    return actual_averages


def plots(all_data, x_axis_numbers, plot_names, test_split=False):
    markers = [".",",","o","v","^","<",">","s","p","*","h","+","x","d","|"]
    numbers = range(10)
    markers.extend(numbers)
    for evaluation_metric in plot_names:
        count = 0
        fig = plt.figure()
        ax = plt.subplot(111)
        for algorithm in all_data.keys():
            if test_split:
                if 'test_split' not in algorithm:
                    continue
            else:
                if 'test_split' in algorithm:
                    continue

            plt.plot(x_axis_numbers, all_data[algorithm][evaluation_metric], label=algorithm, marker=markers[count])
            count += 1
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 6})
        ax.set_title(f"The {evaluation_metric} for all algorithms over different score thresholds")
        ax.set_xlabel("Score threshold")
        ax.set_ylabel(evaluation_metric)
        #plt.legend(loc=(1.04, 0))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()


if __name__ == '__main__':
    scores = np.arange(0.0, 1.0, 0.05)
    scores = range(0, 100, 5)
    #scores = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    final_scores = []
    result = {}
    for score in scores:
        score = score/100
        files = load_files(score)
        if len(files) == 0:
            continue
        result = get_average(files, result)
        final_scores.append(score)

    different_plots = ['precision', 'recall', 'f1', 'bmd', 'variation_of_information']

    #plots(result, scores, different_plots)
    plots(result, final_scores, different_plots, test_split=True)
    print("almost done")

