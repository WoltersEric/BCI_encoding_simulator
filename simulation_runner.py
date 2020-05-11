from simulation import Simulation
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Runner:
    def __init__(self, iterations, interval, noise, algorithms,human_error):
        self.data = []
        self.scanning_speed = interval
        self.algorithms = algorithms
        self.noise = noise
        self.iterations = iterations
        self.human_error = human_error

    def run_simulation(self):
        columns = ['N', 'Algorithm', 'Noise', 'TotalClicks']
        df = pd.DataFrame(columns=columns)
        if self.noise.shape[0] > 1:
            for algor in self.algorithms:
                Algorithm = Simulation("text.txt", algor, self.human_error, iterations=self.iterations)
                for count, i in enumerate(self.noise[0]):
                    for j in self.noise[1]:
                        tmp = Algorithm.simulate(noise=[i, j])
                        df = df.append(tmp, ignore_index=True)
                    print('stap {0}/{1} van {2} klaar'.format(count + 1, len(self.noise[0]), algor))
        else:
            for algor in self.algorithms:
                Algorithm = Simulation("text.txt", algor, iterations=self.iterations)
                for count, i in enumerate(self.noise[0]):
                    tmp = Algorithm.simulate(noise=[i])
                    df = df.append(tmp, ignore_index=True)
                    print('stap {0}/{1} van {2} klaar'.format(count + 1, len(self.noise), algor))
        self.data = df

    def plot_results(self):
        self.data[['TotalClicks']] = self.data[['TotalClicks']].apply(pd.to_numeric)
        # print(self.data.groupby(['Noise', 'Algorithm'])['TotalClicks'].mean())
        # fig, ax = plt.subplots()
        # sns.boxplot(x='Algorithm', y='TotalClicks', hue='Noise', data=self.data, ax=ax)
        # plt.show()
        x_axis = ['TN=0.95:TP={}'.format(x) for x in list(np.around(np.arange(0.75, 0.99, 0.025), decimals=3))]
        fig, ax = plt.subplots()
        sns.lineplot(x='Noise', y='TotalClicks', hue='Algorithm', data=self.data[self.data['Noise'] != 'no noise'], ax=ax)
        ax.set_title('Model results for the sentence: typing')
        ax.set_ylabel('Number of decisions')
        plt.xticks(np.arange(len(x_axis)), x_axis)
        plt.show()

if __name__ == '__main__':
    interval = 2.6
    algorithms = ["Huffman", "RowColumn", "Weighted", "VLEC"]
    # algorithms = ["Huffman", "RowColumn", "VLEC"]
    # algorithms = ["RowColumn"]
    human_error = {'RowColumn': 0.02444444, 'Huffman': 0.03126761, 'VLEC': 0.01882353, 'Weighted': 0.03126761}
    # human_error = {}
    iterations = 100
    # true_positives = list(np.around(np.arange(0.75, 0.99, 0.025), decimals=3))
    # true_negatives = list(np.arange(0.75, 0.99, 0.025))
    true_positives = [0.875]
    true_negatives = [0.95]
    # noise = [[i, j] for i, j in zip(true_negatives,true_positives)]
    noise = [true_negatives, true_positives]
    Run = Runner(iterations, interval, np.array(noise), algorithms, human_error)
    Run.run_simulation()
    Run.plot_results()
    # Run.data.to_pickle("./model_data_extra.p")
    print(Run.data.groupby(['Algorithm', 'Noise'])['TotalClicks'].mean())