import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from src.jguides_2024.utils.plot_helpers import format_ax
from src.jguides_2024.utils.set_helpers import check_membership


class ExponentialKernel:
    def __init__(self,
                 kernel_num_samples,
                 kernel_tau,
                 kernel_center=0,
                 kernel_symmetric=False,
                 kernel_offset=0,
                 density=True):
        self.kernel_parameters = {"kernel_num_samples": kernel_num_samples,
                                  "kernel_tau": kernel_tau,
                                  "kernel_center": kernel_center,
                                  "kernel_symmetric": kernel_symmetric,
                                  "kernel_offset": kernel_offset,
                                  "density": density}
        self._check_inputs()
        self.kernel = self._get_kernel()

    def _check_inputs(self):
        if self.kernel_parameters["kernel_offset"] < 0:
            raise Exception(f"kernel_offset must be nonnegative")

    def _get_kernel(self):
        params = self.kernel_parameters
        kernel = sp.signal.exponential(params["kernel_num_samples"],
                                       params["kernel_center"],
                                       params["kernel_tau"],
                                       params["kernel_symmetric"])
        kernel = np.concatenate(([0] * params["kernel_offset"], kernel))  # offset kernel with zeros
        if params["density"]:  # normalize kernel to sum to one
            kernel /= np.sum(kernel)
        return kernel

    def plot_kernel(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(self.kernel, '.-', color="black")
        format_ax(title="exponential kernel")

    def convolve(self, x, mode, verbose=False):
        convolved_x = np.convolve(x, self.kernel, mode=mode)
        if verbose:
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(x, '.-', label="x")
            ax.plot(convolved_x, '.-', label="convolved_x")
            ax.legend()
        return convolved_x


class Kernel:
    def __init__(self, kernel_params):
        self.kernel_params = kernel_params
        self._check_params()
        self.kernel = self._get_kernel()

    def _check_params(self):
        # If kernel params not empty, check valid
        if len(self.kernel_params) > 0:
            # Check required params passed
            check_membership(["kernel_type"], self.kernel_params)

            # Check kernel type valid
            check_membership([self.kernel_params["kernel_type"]], ["exponential"])

            # Check required params passed for given kernel type
            if self.kernel_params["kernel_type"] == "exponential":
                check_membership(["kernel_num_samples", "kernel_tau"], self.kernel_params)

    def _get_kernel(self):
        # If kernel params empty, return None
        if len(self.kernel_params) == 0:
            return None

        if self.kernel_params["kernel_type"] == "exponential":
            return ExponentialKernel(
                self.kernel_params["kernel_num_samples"], self.kernel_params["kernel_tau"], kernel_center=0,
                kernel_symmetric=False, kernel_offset=1, density=True)
