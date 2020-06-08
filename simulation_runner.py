from simulation import Simulation
import numpy as np
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
        columns = ['N', 'Algorithm', 'Noise', 'TotalClicks', 'NmOfOnes', 'NmOfZeros']
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
        x_axis = ['TP={}%'.format(x) for x in list(np.around(np.arange(0.75, 0.99, 0.025), decimals=3))]
        x_axis[0] = 'TN=0.95%:TP=0.75%'
        fig, ax = plt.subplots()
        sns.lineplot(x='Noise', y='TotalClicks', hue='Algorithm', data=self.data[self.data['Noise'] != 'no noise'], ax=ax)
        ax.set_title('Model results for the sentence: i am a unp user')
        ax.set_ylabel('Number of decisions')
        plt.xticks(np.arange(len(x_axis)), x_axis)
        plt.show()

if __name__ == '__main__':
    algorithms = ["Huffman", "RowColumn", "Weighted", "VLEC"]
    human_error = {'RowColumn': 0.02444444, 'Huffman': 0.03126761, 'VLEC': 0.01882353, 'Weighted': 0.03126761}
    iterations = 1
    interval = 2.6
    true_positives = list(np.around(np.arange(0.75, 0.99, 0.025), decimals=3))
    true_positives = [0.9]
    true_negatives = list(np.around(np.arange(0.75, 0.99, 0.025), decimals=3))
    noise = [true_negatives, true_positives]
    Run = Runner(iterations, interval, np.array(noise), algorithms, human_error)
    Run.run_simulation()