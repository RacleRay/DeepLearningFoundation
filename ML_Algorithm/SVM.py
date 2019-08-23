import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
              1: np.array([[10, 21],
                           [22, 48],
                           [32, 58]])}

class SupportVectorMachine:
    """一个搜索式SVM，用于理解算法，不适用于处理较大数据集"""
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        """train process, to find w and b"""
        self.data = data

        # {||w||: [w, b]}
        opt_dict = {}

        # transpose to every direction that has the same L2 norm
        transfroms = [[1, 1],
                      [1, -1],
                      [-1, 1],
                      [-1, -1]]

        # 以最大、最小的x作为搜索边界，想想||w||的几何表示，这是有意义的
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = float(max(all_data))
        self.min_feature_value = float(min(all_data))
        all_data = None

        # w搜索步长
        step_sizes = [self.max_feature_value * 10**(-scale)\
                         for scale in range(1, 5)]

        b_range_multiple = 2  # b的变化范围scale, 控制边界平移量
        b_multiple = 5        # b更新步长，为w的b_multiple倍
                              # b的更新更直接快速，并且最优解相对容易求得

        # temporary optimal value of w
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # it`s convex
            optimized = False
            while not optimized:  # 线性可分问题
                # 以下b的循环可以并行
                for b in np.arange(-1 * (self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step * b_multiple):
                    for transfromation in transfroms:
                        w_t = w * transfromation
                        found_optional = True
                        # check data，yi*(w.x + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_optional = False

                        if found_optional:
                            # print('found_optional, line75')
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                # transforms保证了在不同方向上的搜索，当小于0，norm是对称的
                if w[0] < 0:
                    optimized = True
                    print('optimized a step')
                else:
                    w -= step

            if len(opt_dict) == 0:
                raise '两类线性不可分'

            # 当前最优值，min（||w||）
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step

        # 检查每个点的值，搜索方法得到近似最优解，sv的值近似于1
        # 可w调整步长，但是搜索方法代价很大  —_—
        for i in data_dict:
            for x in data_dict[i]:
                yi = i
                print(x, ':', (np.dot(self.w, x) + self.b))

    def predict(self, features):
        # sign of w.x + b
        classification = np.sign(np.dot(np.array(features, self.w) + self.b))
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*',
                            c=self.colors[classification])
        return classification

    def visualize(self, data_dict):
        for i in data_dict:
            for x in data_dict[i]:
                self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])

        # v = w.x + b
        # positive support vector: v = 1
        # negative support vector: v = -1
        # boundary: v = 0
        def hyperplane_point(x, w, b, v):
            """在给定一个axis feature value时，计算另一个axis feature value
            即，计算一个在该条线上的坐标"""
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value,
                     self.max_feature_value)

        # x axis range
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # positive support vector
        psv1 = hyperplane_point(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane_point(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        # negative support vector
        nsv1 = hyperplane_point(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane_point(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # decision boundary
        db1 = hyperplane_point(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane_point(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])

        plt.show()


if __name__ == "__main__":
    svm = SupportVectorMachine()
    svm.fit(data_dict)
    svm.visualize(data_dict)