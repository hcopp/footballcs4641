from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


def make_histogram():
    label_file_name = './data/labels.csv'
    labels = read_csv(label_file_name)['Yards'].values
    hist_dict = {}
    for x in labels:
        if x in hist_dict:
            hist_dict[x] += 1
        else:
            hist_dict[x] = 1
    plt.hist(labels, bins=115)
    plt.xticks(range(-20, 101, 5))
    plt.xlabel("Yards")
    plt.ylabel("Number of Occurrences")
    plt.title("Play Results Histogram")
    q75, q25 = np.percentile(labels, [75, 25])
    outliers = [x for x in labels if x < q25 - (5 * 1.5) or x > (5 * 1.5) + q75]
    print(f'Average Yards Gained: {round(np.average(labels), 4)}')
    print(f'Number of Outliers: {len(outliers)}')
    print(f'Percent Outliers: {round(len(outliers)/len(labels)*100, 4)}%')
    plt.savefig('./graphs/label_hist.png')
    plt.show()


if __name__ == '__main__':
    make_histogram()
