from matplotlib import pyplot as plt


class LRCurveVisualizer:
    def __init__(self, calc_metric_per_epoch: int, metric_name: str = None):
        """
        :param calc_metric_per_epoch: 何エポックごとに指標を算出するか
        :param metric_name:
        """
        self.loss_list = []
        self.metric_list = []
        self._metric_name = metric_name
        self._calc_metric_per_epoch = calc_metric_per_epoch

    def add(self, loss: float, metric: float or None = None):
        self.loss_list.append(loss)
        if metric:
            assert isinstance(metric, float) or isinstance(metric, int), "グラフ用Metricはスカラー値である必要がある"
            self.metric_list.append(metric)

    def add_metric(self, metric: float or None = None):
        if metric is None:
            return
        self.metric_list += [metric for _ in range(self._calc_metric_per_epoch)]

    def save_graph_image(self, filepath: str):
        fig, ax1 = plt.subplots()
        x_axis = list(range(len(self.loss_list)))
        loss_line = ax1.plot(x_axis, self.loss_list, label="Loss", color="blue")
        if len(self.metric_list) != 0:
            ax2 = ax1.twinx()
            metric_line = ax2.plot(x_axis, self.metric_list, label=self._metric_name, color="orange")
            lines = metric_line + loss_line
        else:
            lines = loss_line
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc=0)
        plt.savefig(filepath)
        plt.cla()
        plt.clf()
