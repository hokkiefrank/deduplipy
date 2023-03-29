import os
import json
from datetime import datetime


def filters(testrun):
    try:
        fullstring = testrun['changes_description']
        substring = "for smaller training size"
        if substring not in fullstring:
            print(fullstring)
            return False

        substring2 = "no weight"
        if substring2 not in fullstring:
            print(fullstring)
            return False

        if testrun['config']['train_test_split'] != 0.8:
            print(testrun['config']['train_test_split'])
            return False
        if testrun['config']['score_threshold'] != 0.3:
            print(testrun['config']['score_threshold'])
            return False
        if testrun['scored_pairs_table'] != 'scored_pairs_table_musicbrainz20k_full.csv':
            print(testrun['scored_pairs_table'])
            return False
        if testrun['config']['ensemble_cut_probability'] != 0.7:
            return False
        print(f"THIS ONE IS INCLUDED: {fullstring}")
        return True
    except Exception as e:
        print(e)
        return False


def load_files():
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
                if filters(myDict):
                    requesting.append(myDict)
        except Exception as e:
            print(f"This file is bad: {name}")
    print(f"Found: {len(requesting)} files")
    return requesting


def get_average(testruns):
    algorithms = list(testruns[0]['label_dict'].keys())
    algorithms.remove('draw')
    algorithms.append('mixed_best')
    algorithms.append('ensemble_clustering')
    for i in range(len(algorithms)):
        algo = algorithms[i]
        algorithms.append('test_split_'+algo)
    algorithms.append('test_split_predicted_clustering')
    averages = {}
    actual_averages = {}
    for run in testruns:
        for algo in algorithms:
            if algo not in averages:
                averages[algo] = {}
                actual_averages[algo] = {}
            for key, value in run[algo].items():
                if key not in averages[algo]:
                    averages[algo][key] = []
                    actual_averages[algo][key] = 0
                averages[algo][key].append(value)

    for algorithm, scores in averages.items():
        mes = f"{algorithm}:\t"
        for evaluation, values in scores.items():
            average = sum(values)/len(values)
            actual_averages[algorithm][evaluation] = average
            mes += f"{average:.4f}\t"
        print(mes)
    return averages

if __name__ == '__main__':
    files = load_files()
    result = get_average(files)

