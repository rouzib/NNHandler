from torch import nn


class Scaler(nn.Module):
    """
    Performs scaling transformation on input data.

    Scaler is a custom module designed for scaling transformations. It applies
    a linear transformation to the input data based on the provided scaling
    factor and shift value. Optionally, it can also perform the inverse of the
    scaling transformation.

    :ivar m: The shift parameter used in scaling transformation.
    :type m: float
    :ivar c: The scaling factor used in the transformation.
    :type c: float
    :ivar inverse: A flag indicating whether to apply the inverse
        transformation. Defaults to False.
    :type inverse: bool
    """
    def __init__(self, m, c, inverse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m
        self.c = c
        self.inverse = inverse

    def forward(self, x):
        """
        Computes either a forward or inverse transformation based on the value
        of the `inverse` attribute.

        If `inverse` is True, performs the forward transformation by scaling and
        translating the input `x` using the attributes `c` (coefficient) and `m`
        (offset). Otherwise, performs the inverse transformation by reversing
        the scaling and translation.

        :param x: Input value to be transformed.
        :type x: float or int
        :return: Result of the forward or inverse transformation.
        :rtype: float
        """
        if self.inverse:
            return x * self.c + self.m
        return (x - self.m) / self.c
