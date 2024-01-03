import numpy as np
import matplotlib.pyplot as plt

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from data import balance_assist_without_rider, balance_assist_with_rider
from model import SteerControlModel, Meijaard2007Model

from AnalyzeData import TU_COLORS


def main():
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    font = {"size": 16}
    plt.rc("font", **font)

    p = Meijaard2007ParameterSet(balance_assist_with_rider, True)
    m = SteerControlModel(p)
    times = np.linspace(0.0, 5.0, num=501)
    x0 = np.deg2rad([0.0, 0.0, 0.0, 20.0])
    speed = 10 / 3.6
    states_without, _ = m.simulate(times, x0, v=speed)
    states6, _ = m.simulate(times, x0, v=speed, kphidot=controller(6, speed))
    states8, _ = m.simulate(times, x0, v=speed, kphidot=controller(8, speed))
    states10, _ = m.simulate(times, x0, v=speed, kphidot=controller(10, speed))

    roll_angle_without = np.rad2deg(states_without[:, 1])
    roll_angle6 = np.rad2deg(states6[:, 1])
    roll_angle8 = np.rad2deg(states8[:, 1])
    roll_angle10 = np.rad2deg(states10[:, 1])

    for i, ra in enumerate(roll_angle_without):
        if ra < -90:
            break
    roll_angle_without[i:] = -90

    fig, axs = plt.subplots()
    roll_angles = [roll_angle_without, roll_angle6, roll_angle8, roll_angle10]
    colors = ["black", TU_COLORS["blue"], TU_COLORS["red"], TU_COLORS["yellow"]]
    # for ra, color in zip(roll_angles, colors):
    # axs.plot(times, ra, color=color, linewidth=3)

    axs.plot(
        times, roll_angles[0], color="black", linewidth=3, label="Without controller"
    )
    axs.plot(
        times,
        roll_angles[2],
        color=TU_COLORS["blue"],
        linewidth=3,
        label="With controller",
    )
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Steer angle [deg]")
    axs.set_xlim((0, 5))
    axs.set_ylim((-75, 75))
    axs.set_title(
        "Simulation of balance-assist bicycle with rigid rider with initial steer rate of 20 deg/s"
    )
    # axs.legend(["No controller", "Gain -6", "Gain -8", "Gain -10"])
    axs.legend()
    fig.set_size_inches(16, 9)
    # plt.show()
    fig.savefig("controller-simulation", dpi=300)


def controller(gain, speed, vmax=4.7):
    return -gain * (vmax - speed)


if __name__ == "__main__":
    main()
