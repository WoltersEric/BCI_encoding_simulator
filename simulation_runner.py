from simulation import Simulation
import numpy as np
import copy
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, iterations, interval, biased_weight, algorithms):
        self.optimal_result = []
        self.actual_result = []
        self.x_axis = []
        self.scanning_speed = interval
        self.algorithms = algorithms
        self.biased_noise = biased_weight
        self.iterations = iterations

    def run_simulation(self):
        result_optimal = {
            'RowColumn': [],
            'Huffman': [],
            'VLEC': [],
            'Weighted': [],
        }
        result_actual = {
            'RowColumn': [],
            'Huffman': [],
            'VLEC': [],
            'Weighted': [],
        }
        x_axis = list(np.arange(0.75, 0.99, 0.025))
        x_axis = [0.95]
        if self.biased_noise:
            for algor in self.algorithms:
                Algorithm = Simulation("text.txt", algor, iterations=25)
                for count, i in enumerate(x_axis):
                    tmp_list = []
                    for j in x_axis:
                        tmp = Algorithm.simulate(noise=[i, j])
                        tmp_list.append(tmp[1])
                        if (count == len(x_axis) - 1) and (j > 0.97):
                            result_optimal[algor].append(tmp[0])
                    result_actual[algor].append(tmp_list)
                    print('stap {0}/{1} van {2} klaar'.format(count + 1, len(x_axis), algor))
        else:
            for algor in self.algorithms:
                Algorithm = Simulation("text.txt", algor, iterations=100)
                for count, i in enumerate(x_axis):
                    tmp = Algorithm.simulate(noise=[0.875, 0.95])
                    # tmp = Algorithm.simulate(noise=[0.95, 0.875])
                    result_actual[algor].append(tmp[1])
                    if count == len(x_axis) - 1:
                        result_optimal[algor].append(tmp[0])
                    print('stap {0}/{1} van {2} klaar'.format(count + 1, len(x_axis), algor))
        self.x_axis = x_axis
        self.actual_result = result_actual
        self.optimal_result = result_optimal

    def plot_results(self):
        x_axis = self.x_axis
        x_axis = [round(x, 2) for x in x_axis]
        result_optimal = self.optimal_result
        result_actual = self.actual_result
        if self.biased_noise:
            for algor in self.algorithms:
                result = copy.deepcopy(np.array(result_actual[algor]))
                result.astype(float)
                fig, ax = plt.subplots()
                im = ax.imshow(result, cmap='RdBu_r', vmin=0, vmax=90)
                ax.set_xticks(np.arange(len(x_axis)))
                ax.set_yticks(np.arange(len(x_axis)))
                ax.set_xticklabels(x_axis)
                ax.set_yticklabels(x_axis)
                ax.set_title('Performance of {} spelling paradigm in min for 25 iterations'.format(algor))
                ax.set_xlabel('True negative rate')
                ax.set_ylabel('True positive rate')
                fig.colorbar(im, ax=ax)
                fig.set_label('Time to spell full sentence in minutes')
                fig.tight_layout()
                ax.legend()
                plt.show()
        else:
            fig, ax = plt.subplots()
            try:
                horiz_line_datah = np.array([result_optimal['Huffman'] for i in x_axis])
                ax.plot(x_axis, horiz_line_datah, color='red', alpha=0.4)
                y_axish = np.array(result_actual['Huffman'])
                y_axish[y_axish == 0] = 'nan'
                ax.plot(x_axis, y_axish, color='red', alpha=0.7, label="Huffman paradigm")
            except:
                pass

            try:
                horiz_line_datar = np.array([result_optimal['RowColumn'] for i in x_axis])
                ax.plot(x_axis, horiz_line_datar, color='blue', alpha=0.4)
                y_axisr = np.array(result_actual['RowColumn'])
                y_axisr[y_axisr == 0] = 'nan'
                ax.plot(x_axis, y_axisr, color='blue', alpha=0.7, label="RowColumn paradigm")
            except:
                pass

            try:
                horiz_line_datav = np.array([result_optimal['VLEC'] for i in x_axis])
                ax.plot(x_axis, horiz_line_datav, color='black', alpha=0.4)
                y_axisv = np.array(result_actual['VLEC'])
                y_axisv[y_axisv == 0] = 'nan'
                ax.plot(x_axis, y_axisv, color='black', alpha=0.7, label="VLEC paradigm")
            except:
                pass

            try:
                horiz_line_datav = np.array([result_optimal['Weighted'] for i in x_axis])
                ax.plot(x_axis, horiz_line_datav, color='green', alpha=0.4)
                y_axisv = np.array(result_actual['Weighted'])
                y_axisv[y_axisv == 0] = 'nan'
                ax.plot(x_axis[3:], y_axisv[3:], color='green', alpha=0.7, label="Weighted paradigm")
            except:
                pass

            ax.set_xlabel('Click accuracy')
            ax.set_ylabel('Time in minutes')
            ax.set_title('Performance of different spelling paradigms with biased noise [0.8, x] for 30 iterations')
            ax.legend()
            plt.show()

if __name__ == '__main__':
    interval = 2.6
    biased_weight = False
    algorithms = ["Huffman", "RowColumn", "Weighted", "VLEC"]
    algorithms = ["Huffman", "RowColumn", "VLEC"]
    iterations = 5
    Run = Runner(iterations, interval, biased_weight, algorithms)
    Run.run_simulation()
    Run.plot_results()