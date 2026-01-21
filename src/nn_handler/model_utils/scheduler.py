from abc import ABC, abstractmethod
import math

class Schedule(ABC):
    """
    Abstract base class for defining schedules.

    This class serves as a base for creating custom schedules. A schedule is a
    function or mapping that determines values based on progress through epochs.
    Subclasses must implement the `get_value` method to specify how the value is
    computed for a given epoch. The `plot` method can be used for visualizing the
    schedule's values over a range of epochs.

    :ivar attribute1: This class does not define attributes, but subclasses may
        define specific attributes (e.g., to store parameters controlling the
        schedule). This section is illustrative.
    :type attribute1: Any
    """
    @abstractmethod
    def get_value(self, epoch: int) -> float:
        """
        An abstract method to compute and retrieve a specific float value based on
        the given epoch. Subclasses are required to provide an implementation for
        this method to ensure consistent behavior across different use cases.

        :param epoch: The current epoch for which the value needs to be calculated.
                      It is typically a non-negative integer representing a step
                      or iteration in a process.
        :type epoch: int

        :return: A floating-point value derived from the provided epoch.
        :rtype: float
        """
        pass

    def plot(self, epochs: int = 100, title: str = "Schedule"):
        """
        Plots the schedule over a specified number of epochs using Matplotlib.

        This method generates a line plot illustrating the values computed by the
        `schedule` instance across multiple epochs. It requires the Matplotlib
        library to function and will raise an ImportError if not available. The user
        can customize the number of epochs and the title of the plot.

        :param epochs: Number of epochs for which the schedule should be plotted.
                       Defaults to 100.
        :type epochs: int
        :param title: Title of the plot. Defaults to "Schedule".
        :type title: str
        :return: None
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required to plot the schedule.")

        x = list(range(epochs))
        y = [self.get_value(e) for e in x]
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()


class LogLinearSchedule(Schedule):
    """
    Represents a logarithmic linear schedule for adjusting values over epochs.

    Provides a flexible schedule where values are interpolated on a logarithmic
    scale between a starting and ending point over a specified range of epochs.

    :ivar start_value: The initial value of the schedule.
    :type start_value: float
    :ivar end_value: The final value of the schedule.
    :type end_value: float
    :ivar start_epoch: The epoch at which the schedule begins.
    :type start_epoch: int
    :ivar end_epoch: The epoch at which the schedule ends.
    :type end_epoch: int
    """
    def __init__(self, start_value: float, end_value: float, start_epoch: int, end_epoch: int):
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def get_value(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return self.start_value
        elif epoch >= self.end_epoch:
            return self.end_value
        else:
            log_start = math.log(self.start_value)
            log_end = math.log(self.end_value)
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return math.exp(log_start + progress * (log_end - log_start))
