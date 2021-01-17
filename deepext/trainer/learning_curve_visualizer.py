from matplotlib import pyplot as plt


class LearningCurveVisualizer:
    def __init__(self, metric_name: str, save_filepath: str,
                 ignore_epoch=0):
        """
        :param metric_name:
        :param save_filepath: グラフ画像保存先ファイルパス
        :param ignore_epoch: 最初に何エポック無視するか
        """
        self.loss_list = []
        self.metric_test_list = []
        self.metric_train_list = []
        self._metric_name = metric_name
        self._save_filepath = save_filepath
        self._ignore_epoch = ignore_epoch

    def add_loss(self, loss: float):
        assert isinstance(loss, float) or isinstance(loss, int), "Lossはスカラー値である必要がある"
        self.loss_list.append(loss)

    def add_metrics(self, train_metric: float or None = None, test_metric: float or None = None,
                    calc_metric_per_epoch: int = 1):
        if train_metric is not None:
            assert isinstance(train_metric, float) or isinstance(train_metric, int), "グラフ用Metricはスカラー値である必要がある"
            self.metric_train_list += [train_metric for _ in range(calc_metric_per_epoch)]
        if test_metric is not None:
            assert isinstance(test_metric, float) or isinstance(test_metric, int), "グラフ用Metricはスカラー値である必要がある"
            self.metric_test_list += [test_metric for _ in range(calc_metric_per_epoch)]

    def save_graph_image(self):
        if len(self.loss_list) <= self._ignore_epoch:
            return
        fig, base_ax = plt.subplots()
        x_axis = list(range(len(self.loss_list)))[self._ignore_epoch:]
        loss_line = base_ax.plot(x_axis, self.loss_list[self._ignore_epoch:], label="Loss", color="blue")
        lines = loss_line
        metric_ax = base_ax.twinx()
        if len(self.metric_test_list) != 0:
            metric_lines = metric_ax.plot(x_axis, self.metric_test_list[self._ignore_epoch:],
                                          label="Test " + self._metric_name,
                                          color="orange")
            lines += metric_lines
        if len(self.metric_train_list) != 0:
            metric_lines = metric_ax.plot(x_axis, self.metric_train_list[self._ignore_epoch:],
                                          label="Train " + self._metric_name,
                                          color="lightblue")
            lines += metric_lines

        labels = [l.get_label() for l in lines]
        base_ax.legend(lines, labels, loc=0)
        plt.savefig(self._save_filepath)
        plt.cla()
        plt.clf()
        plt.close()
