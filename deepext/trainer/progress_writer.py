from abc import ABCMeta, abstractmethod
from typing import List


class ProgressWriter(metaclass=ABCMeta):
    @abstractmethod
    def out_training_start(self, config: str) -> 'ProgressWriter':
        pass

    @abstractmethod
    def out_epoch(self, epoch: int, max_epoch: int, elapsed_time: float, mean_loss: float) -> 'ProgressWriter':
        pass

    @abstractmethod
    def out_elapsed_time(self, elapsed_time: float) -> 'ProgressWriter':
        pass

    @abstractmethod
    def out_metrics(self, metrics_name: str, metrics_val: str, prefix: str) -> 'ProgressWriter':
        pass

    @abstractmethod
    def out_heading(self, title: str) -> 'ProgressWriter':
        pass

    @abstractmethod
    def out_divider(self) -> 'ProgressWriter':
        pass

    @abstractmethod
    def out_small_divider(self) -> 'ProgressWriter':
        pass


class StdOutProgressWriter(ProgressWriter):
    def out_training_start(self, config: str):
        print(f"\n\nStart training:  {config}\n\n")
        return self

    def out_epoch(self, epoch: int, max_epoch: int, elapsed_time: float, mean_loss: float):
        print(f"epoch {epoch + 1} / {max_epoch} :  {elapsed_time}s   --- loss: {mean_loss}")
        return self

    def out_elapsed_time(self, elapsed_time: float):
        print(f"Elapsed time: {elapsed_time}s")
        return self

    def out_metrics(self, metrics_name, metrics_val, prefix: str):
        print(f"{prefix} {metrics_name} : {metrics_val}")
        return self

    def out_heading(self, title: str) -> 'ProgressWriter':
        print(f"\n{title}\n")
        return self

    def out_divider(self):
        print("\n\n\n\n")
        return self

    def out_small_divider(self):
        print("\n\n")
        return self


class FileOutputProgressWriter(ProgressWriter):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def out_training_start(self, config: str) -> 'ProgressWriter':
        self._write_file(f"\n\nStart training:  {config}\n\n")
        return self

    def out_epoch(self, epoch: int, max_epoch: int, elapsed_time: float, mean_loss: float) -> 'ProgressWriter':
        self._write_file(f"epoch {epoch + 1} / {max_epoch} :  {elapsed_time}s   --- loss: {mean_loss}")
        return self

    def out_elapsed_time(self, elapsed_time: float) -> 'ProgressWriter':
        self._write_file(f"Elapsed time: {elapsed_time}s")
        return self

    def out_metrics(self, metrics_name: str, metrics_val: str, prefix: str) -> 'ProgressWriter':
        self._write_file(f"{prefix} {metrics_name} : {metrics_val}")
        return self

    def out_heading(self, title: str) -> 'ProgressWriter':
        self._write_file(f"\n{title}\n")
        return self

    def out_divider(self) -> 'ProgressWriter':
        self._write_file("\n\n\n\n")
        return self

    def out_small_divider(self) -> 'ProgressWriter':
        self._write_file("\n\n")
        return self

    def _write_file(self, val: str):
        with open(self._filepath, "a") as file:
            file.write(val + "\n")


class CSVProgressWriter(ProgressWriter):
    """
    col1: type
    col2: value
    col3: detail value(unnecessary)
    """

    def __init__(self, filepath: str):
        self._filepath = filepath

    def out_training_start(self, config: str) -> 'ProgressWriter':
        self._write_file(["Start", config, ""])
        return self

    def out_epoch(self, epoch: int, max_epoch: int, elapsed_time: float, mean_loss: float) -> 'ProgressWriter':
        self._write_file([f"epoch{epoch}", mean_loss, elapsed_time])
        return self

    def out_elapsed_time(self, elapsed_time: float) -> 'ProgressWriter':
        self._write_file(["time", elapsed_time, ""])
        return self

    def out_metrics(self, metrics_name: str, metrics_val: str, prefix: str) -> 'ProgressWriter':
        self._write_file([f"{prefix}metrics", metrics_val, metrics_name])
        return self

    def out_heading(self, title: str) -> 'ProgressWriter':
        self._write_file(["heading", title, ""])
        return self

    def out_divider(self) -> 'ProgressWriter':
        pass

    def out_small_divider(self) -> 'ProgressWriter':
        pass

    def _write_file(self, val_ls: List[str]):
        with open(self._filepath, "a") as file:
            file.write(",".join(val_ls) + "\n")


class MultiProgressWriter(ProgressWriter):
    def __init__(self, writers: List[ProgressWriter]):
        self._writers = writers

    def out_training_start(self, config: str) -> 'ProgressWriter':
        for writer in self._writers:
            writer.out_training_start(config)
        return self

    def out_epoch(self, epoch: int, max_epoch: int, elapsed_time: float, mean_loss: float) -> 'ProgressWriter':
        for writer in self._writers:
            writer.out_epoch(epoch, max_epoch, elapsed_time, mean_loss)
        return self

    def out_elapsed_time(self, elapsed_time: float) -> 'ProgressWriter':
        for writer in self._writers:
            writer.out_elapsed_time(elapsed_time)
        return self

    def out_metrics(self, metrics_name: str, metrics_val: str, prefix: str) -> 'ProgressWriter':
        for writer in self._writers:
            writer.out_metrics(metrics_name, metrics_val, prefix)
        return self

    def out_heading(self, title: str) -> 'ProgressWriter':
        for writer in self._writers:
            writer.out_heading(title)
        return self

    def out_divider(self) -> 'ProgressWriter':
        for writer in self._writers:
            writer.out_divider()
        return self

    def out_small_divider(self) -> 'ProgressWriter':
        for writer in self._writers:
            writer.out_small_divider()
        return self
