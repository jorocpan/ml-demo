import math
import random

random.seed(0)


def rand(a, b):
    """
    生成 a 到 b 之间的随机数
    :param a:
    :param b:
    :return:
    """
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    """
    构建矩阵 m 行 n 列
    :param m:
    :param n:
    :param fill:
    :return:
    """
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    print(mat)
    return mat


def sigmoid(x):
    """
    激活函数
    :param x:
    :return:
    """
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(y):
    """
    激活函数的导函数
    :param y:
    :return:
    """
    return y * (1 - y)


class BPNeuralNetwork:
    def __init__(self):
        """
        构建网络基数
        """
        self.input_number = 0
        self.hidden_number = 0
        self.output_number = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, number_of_input, number_of_hide, number_of_output):
        self.input_number = number_of_input + 1
        self.hidden_number = number_of_hide
        self.output_number = number_of_output
        # init cells
        self.input_cells = [1.0] * self.input_number
        self.hidden_cells = [1.0] * self.hidden_number
        self.output_cells = [1.0] * self.output_number
        # init weights
        self.input_weights = make_matrix(self.input_number, self.hidden_number)
        self.output_weights = make_matrix(self.hidden_number, self.output_number)
        # random activate
        for i in range(self.input_number):
            for h in range(self.hidden_number):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_number):
            for o in range(self.output_number):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_number, self.hidden_number)
        self.output_correction = make_matrix(self.hidden_number, self.output_number)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_number - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_number):
            total = 0.0
            for i in range(self.input_number):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_number):
            total = 0.0
            for j in range(self.hidden_number):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_number
        for o in range(self.output_number):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_number
        for h in range(self.hidden_number):
            error = 0.0
            for o in range(self.output_number):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_number):
            for o in range(self.output_number):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_number):
            for h in range(self.hidden_number):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)
        self.train(cases, labels, 10000, 0.05, 0.1)
        for case in cases:
            print(self.predict(case))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()