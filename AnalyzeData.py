import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy.optimize as spo

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from data import (
    balance_assist_without_rider,
    benchmark,
    balance_assist_with_rider,
    rigid_bike_without_rider,
    rigid_bike_with_rider,
)
from model import SteerControlModel, Meijaard2007Model


def main():
    parameter_set_bas = Meijaard2007ParameterSet(rigid_bike_without_rider, False)
    speeds = np.linspace(0.0, 6.0, num=61)
    velocities = [6, 8, 10, 12, 14, 16, 18]
    velocities_ms = [vel / 3.6 for vel in velocities]
    colors_measured = ["cornflowerblue", "blue", "darkblue"]
    colors_theoretical = ["lightgray", "gray", "black"]

    # plot benchmark + controller and measured
    parameter_set_benchmark = Meijaard2007ParameterSet(benchmark, False)
    model_benchmark_control = SteerControlModel(parameter_set_benchmark)
    fig = plot_measured_eigenvalues(
        speeds,
        model_benchmark_control,
        colors_theoretical,
        colors_measured,
        velocities_ms,
    )
    fig.set_size_inches(8, 6)
    # fig.show()
    fig.suptitle(
        "Theoretical eigenvalues of riderless benchmark bicycle with controller compared to measured eigenvalues at gain of -6, -8 and -10",
        wrap=True,
    )
    plt.savefig("figures/all-gains-benchmark-control-no-rider", dpi=300)
    # fig.waitforbuttonpress()

    # plot theoretical balance-assist with controller and measured
    model_bas_control = SteerControlModel(parameter_set_bas)
    fig = plot_measured_eigenvalues(
        speeds, model_bas_control, colors_theoretical, colors_measured, velocities_ms
    )
    fig.set_size_inches(8, 6)
    # fig.show()
    fig.suptitle(
        "Theoretical eigenvalues of riderless balance-assist bicycle with controller compared to measured eigenvalues at gain of -6, -8 and -10",
        wrap=True,
    )
    plt.savefig("figures/all-gains-bas-control-no-rider", dpi=300)
    # fig.waitforbuttonpress()

    # plot theoretical balance-assist without controller
    model_bas_normal = Meijaard2007Model(parameter_set_bas)
    fig, ax = plt.subplots()
    ax = model_bas_normal.plot_eigenvalue_parts(
        ax=ax,
        v=speeds,
        colors=[
            colors_theoretical[-1],
            colors_theoretical[-1],
            colors_theoretical[-1],
            colors_theoretical[-1],
        ],
    )
    legend_elements = []
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color=colors_theoretical[-1],
            lw=4,
            label="Real",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color=colors_theoretical[-1],
            linestyle="--",
            lw=4,
            label="Imaginary",
        )
    )
    fig.set_size_inches(8, 6)
    ax.set_xlabel("velocity [m/s]")
    plt.legend(handles=legend_elements)
    # fig.show()
    fig.suptitle(
        "Theoretical eigenvalues of riderless balance-assist bicycle without controller",
        wrap=True,
    )
    plt.savefig("figures/theoretical-bas-no-control-no-rider", dpi=300)
    # fig.waitforbuttonpress()

    # plot theoretical balance-assist with rider without controller
    parameter_set_bas_rider = Meijaard2007ParameterSet(rigid_bike_with_rider, True)
    model_bas_rider = Meijaard2007Model(parameter_set_bas_rider)
    fig, ax = plt.subplots()
    ax = model_bas_rider.plot_eigenvalue_parts(
        ax=ax,
        v=speeds,
        colors=[
            colors_theoretical[-1],
            colors_theoretical[-1],
            colors_theoretical[-1],
            colors_theoretical[-1],
        ],
    )
    fig.set_size_inches(8, 6)
    # plt.legend(handles=legend_elements)
    # # fig.show()
    # fig.suptitle(
    #     "Theoretical eigenvalues of balance-assist bicycle with rigid rider and without controller",
    #     wrap=True,
    # )
    # plt.savefig("figures/theoretical-bas-no-control-with-rider", dpi=300)
    # # fig.waitforbuttonpress()

    # plot balance assist with rider with controller
    parameter_set_bas_rider = Meijaard2007ParameterSet(rigid_bike_with_rider, True)
    model_bas_rider_control = SteerControlModel(parameter_set_bas_rider)
    for i, gain in enumerate([6, 8, 10]):
        speed, kphi, kphidots = controller(gain)
        ax = model_bas_rider_control.plot_eigenvalue_parts(
            ax=ax,
            v=speed,
            kphi=kphi,
            kphidot=kphidots,
            colors=[
                colors_measured[i],
                colors_measured[i],
                colors_measured[i],
                colors_measured[i],
            ],
        )
    fig.set_size_inches(8, 6)
    legend_elements = [
        Line2D([0], [0], color=colors_theoretical[-1], lw=4, label="Without controller")
    ]
    gains = ["-6", "-8", "-10"]
    for i, colors in enumerate(zip(colors_theoretical, colors_measured)):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=colors[1],
                lw=4,
                label="Gain " + gains[i],
            )
        )
    plt.legend(handles=legend_elements)
    # fig.show()
    ax.grid()
    ax.set_xlabel("velocity [m/s]")
    fig.suptitle(
        "Theoretical eigenvalues of balance-assist bicycle with rigid rider",
        wrap=True,
    )
    plt.savefig("figures/theoretical-bas-with-control-with-rider", dpi=300)
    # fig.waitforbuttonpress()


def plot_measured_eigenvalues(
    speeds, model, colors_theoretical, colors_measured, velocities_ms
):
    fig, ax = plt.subplots()
    for i, gain in enumerate([6, 8, 10]):
        if gain == 6:
            prefix = "12-May-2023-10-11-04-bas-on-gain-6-speed-"
        else:
            prefix = "12-May-2023-10-39-32-bas-on-gain-" + str(gain) + "-speed-"
        real, img, r_squared = fit_curve(gain, prefix, show_plots=False)

        speed, kphi, kphidots = controller(gain)
        ax = model.plot_eigenvalue_parts(
            ax=ax,
            v=speed,
            kphi=kphi,
            kphidot=kphidots,
            colors=[
                colors_theoretical[i],
                colors_theoretical[i],
                colors_theoretical[i],
                colors_theoretical[i],
            ],
        )
        print(real)
        plt.plot(
            velocities_ms,
            real,
            ".",
            label="Measured real eigenvalues",
            color=colors_measured[i],
        )
        plt.plot(
            velocities_ms,
            img,
            ".",
            label="Measured imaginary eigenvalues",
            color=colors_measured[i],
        )
        plt.legend()
        print(
            f"R-squared values for gain {gain}: Mean: {np.mean(r_squared)}, Max: {np.max(r_squared)}, Min: {np.min(r_squared)}"
        )

    legend_elements = []
    gains = ["-6", "-8", "-10"]
    for i, colors in enumerate(zip(colors_theoretical, colors_measured)):
        legend_elements.append(
            Line2D(
                [0], [0], color=colors[0], lw=4, label="Theoretical, gain " + gains[i]
            )
        ),
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor=colors[1],
                color="w",
                lw=4,
                label="Measured, gain " + gains[i],
            )
        )

    plt.legend(handles=legend_elements)
    ax.set_xlabel("velocity [m/s]")
    return fig


def fit_curve(gain: int, prefix: str, show_plots: bool = False):
    dir = os.path.join("finalized_logs", "bas-on-gain-" + str(gain))
    number_of_files = 3

    velocities = [6, 8, 10, 12, 14, 16, 18]
    avg_eigenvalues_real = []
    avg_eigenvalues_img = []
    r_squared = []

    for j, vel in enumerate(velocities):
        eigenvalues_real = []
        eigenvalues_img = []
        for i in np.linspace(1, number_of_files, number_of_files, dtype=int):
            path = os.path.join(dir, prefix + str(vel))
            df_gyro = pd.read_csv(path + "-gyro-" + str(i) + ".csv")
            popt, pcov = spo.curve_fit(
                kooijman_func,
                df_gyro.loc[:, "time"],
                df_gyro.loc[:, "gyro_x"],
                p0=(-1.0, -1.0, 1.0, 3.0, 1.0),
            )
            eigenvalues_real.append(popt[1])
            eigenvalues_img.append(popt[3])

            # Calculate r-squared
            residuals = df_gyro.loc[:, "gyro_x"] - kooijman_func(
                df_gyro.loc[:, "time"], *popt
            )
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum(
                (df_gyro.loc[:, "gyro_x"] - np.mean(df_gyro.loc[:, "gyro_x"])) ** 2
            )
            r_squared.append(1 - (ss_res / ss_tot))

            if show_plots:
                print(popt)
                fig, ax = plt.subplots(1, 1)
                ax.plot(
                    df_gyro.loc[:, "time"],
                    df_gyro.loc[:, "gyro_x"],
                    ".",
                    label="Measured data",
                )
                ax.plot(
                    df_gyro.loc[:, "time"],
                    kooijman_func(
                        df_gyro.loc[:, "time"],
                        popt[0],
                        popt[1],
                        popt[2],
                        popt[3],
                        popt[4],
                    ),
                    label="Fitted equation",
                )
                ax.set_xlabel("Time since perturbation [s]")
                ax.set_ylabel("Roll rate [rad/s]")
                ax.set_title("Measured roll rate data and fitted equation")
                ax.grid(True)
                ax.legend()
                # fig.show()
                fig.waitforbuttonpress()

        avg_eigenvalues_real.append(sum(eigenvalues_real) / np.size(eigenvalues_real))
        avg_eigenvalues_img.append(sum(eigenvalues_img) / np.size(eigenvalues_img))

    return avg_eigenvalues_real, avg_eigenvalues_img, r_squared


def kooijman_func(t, c1, d, c2, omega, c3):
    return c1 + np.exp(d * t) * (c2 * np.cos(omega * t) + c3 * np.sin(omega * t))


def controller(kv: int, kc: float = -0.7, vmin: float = 1.5, vmax: float = 4.7):
    speeds = np.linspace(0.0, 6.0, num=61)
    kphi = np.ones(speeds.shape)
    kphidots = np.ones(speeds.shape)
    for i, speed in enumerate(speeds):
        if speed < vmin:
            kphidots[i] = -kv * ((vmax - vmin) / vmin) * speed
        if speed <= vmax:
            kphidots[i] = -kv * (vmax - speed)
        elif speed > vmax:
            kphi[i] = kc * (speed - vmax)

    return speeds, kphi, kphidots


if __name__ == "__main__":
    main()
