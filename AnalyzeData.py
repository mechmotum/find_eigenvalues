import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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

TU_COLORS = {
    "blue": (0 / 255, 166 / 255, 214 / 255),
    "red": (224 / 255, 60 / 255, 49 / 255),
    "orange": (237 / 255, 104 / 255, 66 / 255),
    "yellow": (255 / 255, 184 / 255, 28 / 255),
}


def main():
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    font = {"size": 16}
    plt.rc("font", **font)
    mpl.rcParams["lines.linewidth"] = 3

    parameter_set_bas = Meijaard2007ParameterSet(balance_assist_without_rider, False)
    speeds = np.linspace(0.0, 6.0, num=61)
    velocities = [6, 8, 10, 12, 14, 16, 18]
    velocities_ms = [vel / 3.6 for vel in velocities]
    colors_measured = [TU_COLORS["blue"], TU_COLORS["blue"], TU_COLORS["yellow"]]
    colors_theoretical = ["lightgray", "gray", "black"]

    # plot benchmark + controller and measured
    # parameter_set_benchmark = Meijaard2007ParameterSet(benchmark, False)
    # model_benchmark_control = SteerControlModel(parameter_set_benchmark)
    # fig = plot_measured_eigenvalues(
    #     speeds,
    #     model_benchmark_control,
    #     colors_theoretical,
    #     colors_measured,
    #     velocities_ms,
    # )
    # fig.set_size_inches(16, 9)
    # fig.suptitle(
    #     "Theoretical eigenvalues of riderless benchmark bicycle with controller compared to measured eigenvalues at gain of -6, -8 and -10",
    #     wrap=True,
    # )
    # plt.savefig("figures/all-gains-benchmark-control-no-rider", dpi=300)

    # plot theoretical balance-assist with controller and measured
    model_bas_control = SteerControlModel(parameter_set_bas)
    plot_measured_eigenvalues(
        speeds, model_bas_control, colors_theoretical, colors_measured, velocities_ms
    )

    # plot theoretical balance-assist without controller
    speeds10 = np.linspace(0.0, 10.0, num=101)
    model_bas_normal = Meijaard2007Model(parameter_set_bas)
    fig, ax = plt.subplots()
    ax = model_bas_normal.plot_eigenvalue_parts(
        ax=ax,
        v=speeds10,
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
    fig.set_size_inches(16, 9)
    ax.set_xlabel("velocity [m/s]")
    ax.set_ylim(-7.5, 7.5)
    ax.set_ylabel("Magnitude of eigenvalue [1/s]")
    plt.legend(handles=legend_elements)
    # fig.show()
    fig.suptitle(
        "Theoretical eigenvalues of riderless balance-assist bicycle without controller",
        wrap=True,
    )
    plt.savefig("figures/theoretical-bas-no-control-no-rider", dpi=300)
    # fig.waitforbuttonpress()

    # plot theoretical balance-assist with rider without controller
    parameter_set_bas_rider = Meijaard2007ParameterSet(balance_assist_with_rider, True)
    model_bas_rider = Meijaard2007Model(parameter_set_bas_rider)
    fig, ax = plt.subplots()
    ax = model_bas_rider.plot_eigenvalue_parts(
        ax=ax,
        v=speeds10,
        colors=[
            colors_theoretical[-1],
            colors_theoretical[-1],
            "blue",
            "blue",
        ],
    )
    fig.set_size_inches(16, 9)
    for line in ax.get_lines():
        if line.get_color() == "blue":
            line.remove()
    # plt.legend(handles=legend_elements)
    # # fig.show()
    # fig.suptitle(
    #     "Theoretical eigenvalues of balance-assist bicycle with rigid rider and without controller",
    #     wrap=True,
    # )
    # plt.savefig("figures/theoretical-bas-no-control-with-rider", dpi=300)
    # # fig.waitforbuttonpress()

    # plot balance assist with rider with controller
    parameter_set_bas_rider = Meijaard2007ParameterSet(balance_assist_with_rider, True)
    model_bas_rider_control = SteerControlModel(parameter_set_bas_rider)
    for i, gain in enumerate([8]):
        speed, kphi, kphidots = controller(speeds, gain)
        ax = model_bas_rider_control.plot_eigenvalue_parts(
            ax=ax,
            v=speeds,
            kphi=kphi,
            kphidot=kphidots,
            colors=[
                colors_measured[i],
                colors_measured[i],
                "blue",
                "blue",
            ],
        )
        for line in ax.get_lines():
            if line.get_color() == "blue":
                line.remove()

    ax.plot(speeds, np.zeros(speeds.shape), color="black", linewidth=1.5, zorder=0)

    fig.set_size_inches(16, 9)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors_theoretical[-1],
            lw=4,
            label="Real eigenvalues, without controller",
        ),
        Line2D(
            [0],
            [0],
            color=colors_theoretical[-1],
            lw=4,
            linestyle="--",
            label="Imaginary eigenvalues, without controller",
        ),
    ]
    gains = ["-6", "-8", "-10"]
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color=colors_measured[0],
            lw=4,
            label="Real eigenvalues, with controller",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            color=colors_measured[0],
            lw=4,
            linestyle="--",
            label="Imaginary eigenvalues, with controller",
        )
    )
    plt.legend(handles=legend_elements)

    # fig.show()
    ax.grid()
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylim(-7.5, 7.5)
    ax.set_ylabel("Magnitude of eigenvalue [1/s]")
    ax.set_title(
        "Weave mode of the balance-assist bicycle model with rigid rider, with and without controller",
        wrap=True,
    )
    plt.savefig("figures/theoretical-bas-with-control-with-rider", dpi=300)
    # fig.waitforbuttonpress()


def plot_measured_eigenvalues(
    speeds, model, colors_theoretical, colors_measured, velocities_ms
):
    legend_elements = []
    gains = ["-6", "-8", "-10"]
    marker_shapes = ["o", "^", "s"]

    for i, gain in enumerate([6, 8, 10]):
        if gain == 6:
            prefix = "12-May-2023-10-11-04-bas-on-gain-6-speed-"
        else:
            prefix = "12-May-2023-10-39-32-bas-on-gain-" + str(gain) + "-speed-"
        real, img, r_squared = fit_curve(gain, prefix, show_plots=False)
        fig, ax = plt.subplots()
        ax.set_xlabel("velocity [m/s]")
        ax.set_ylabel("Magnitude of eigenvalue [1/s]")
        ax.set_ylim(-12.5, 12.5)
        fig.set_size_inches(16, 9)
        ax.set_title(
            "Theoretical eigenvalues of riderless balance-assist bicycle with controller compared to measured eigenvalues",
            wrap=True,
        )
        speed, kphi, kphidots = controller(speeds, gain)
        ax = model.plot_eigenvalue_parts(
            ax=ax,
            v=speed,
            kphi=kphi,
            kphidot=kphidots,
            colors=[
                "black",
                "black",
                "black",
                "black",
            ],
        )
        for line in ax.get_lines():
            if line.get_linestyle() == "--":
                line.remove()

        legend_elements = []
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=4,
                label="Real eigenvalues of the model",
            )
        )
        ax.legend(handles=legend_elements, loc="lower left")

        plt.savefig("figures/eigenvalues-model-real-" + str(gain) + ".png", dpi=300)

        plt.plot(
            velocities_ms,
            real,
            marker_shapes[i],
            label="Measured real eigenvalues",
            color=colors_measured[i],
            markersize=12,
        )

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker_shapes[i],
                markerfacecolor=colors_measured[i],
                color="w",
                markersize=15,
                label="Measured real eigenvalues",
            )
        )
        ax.legend(handles=legend_elements, loc="lower left")

        plt.savefig(
            "figures/eigenvalues-model-and-measured-real-" + str(gain) + ".png", dpi=300
        )

        for line in ax.get_lines():
            line.remove()

        ax = model.plot_eigenvalue_parts(
            ax=ax,
            v=speed,
            kphi=kphi,
            kphidot=kphidots,
            colors=[
                "black",
                "black",
                "black",
                "black",
            ],
        )
        ax.grid()

        plt.plot(
            velocities_ms,
            real,
            marker_shapes[i],
            label="Measured real eigenvalues",
            color=colors_measured[i],
            markersize=12,
        )

        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=4,
                linestyle="--",
                label="Imaginary eigenvalues of the model",
            )
        )
        ax.legend(handles=legend_elements, loc="lower left")

        plt.savefig(
            "figures/eigenvalues-model-and-measured-real-and-imaginary"
            + str(gain)
            + ".png",
            dpi=300,
        )

        plt.plot(
            velocities_ms,
            img,
            marker_shapes[0],
            label="Measured imaginary eigenvalues",
            color=colors_measured[i],
            markersize=12,
        )

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker_shapes[0],
                markerfacecolor=colors_measured[i],
                color="w",
                markersize=15,
                label="Measured imaginary eigenvalues",
            )
        )

        ax.legend(handles=legend_elements, loc="lower left")
        # for line in ax.get_lines():
        # if line.get_linestyle() == "--":
        #     line.remove()
        # if line.get_color() == "black":
        #     line.remove()
        ax.set_xlabel("Speed [m/s]")
        print(
            f"R-squared values for gain {gain}: Mean: {np.mean(r_squared)}, Max: {np.max(r_squared)}, Min: {np.min(r_squared)}"
        )

        plt.savefig(
            "figures/all-gains-bas-control-no-rider-gain-" + str(gain) + ".png", dpi=300
        )


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
                fig, ax = plt.subplots(1, 1)
                ax.plot(
                    df_gyro.loc[:, "time"],
                    df_gyro.loc[:, "gyro_x"],
                    ".",
                    label="Measured data",
                    color="black",
                    zorder=10,
                )
                ax.set_xlabel("Time since perturbation [s]")
                ax.set_ylabel("Roll rate [rad/s]")
                ax.set_title("Example of measured roll rate data")
                ax.legend()

                fig.set_size_inches(16, 9)
                fig.savefig(
                    "figures/roll_rate_fits/gain-"
                    + str(gain)
                    + "-vel-"
                    + str(vel)
                    + "-file-"
                    + str(i)
                    + "-measured.png",
                    dpi=300,
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
                    color=TU_COLORS["blue"],
                    zorder=5,
                )
                ax.set_title("Example of measured roll rate data and fitted equation")
                ax.legend()

                fig.set_size_inches(16, 9)
                fig.set_dpi(300)
                fig.savefig(
                    "figures/roll_rate_fits/gain-"
                    + str(gain)
                    + "-vel-"
                    + str(vel)
                    + "-file-"
                    + str(i)
                    + "fitted.png",
                    dpi=300,
                )

        avg_eigenvalues_real.append(sum(eigenvalues_real) / np.size(eigenvalues_real))
        avg_eigenvalues_img.append(sum(eigenvalues_img) / np.size(eigenvalues_img))

    return avg_eigenvalues_real, avg_eigenvalues_img, r_squared


def kooijman_func(t, c1, d, c2, omega, c3):
    return c1 + np.exp(d * t) * (c2 * np.cos(omega * t) + c3 * np.sin(omega * t))


def controller(speeds, kv: int, kc: float = -0.7, vmin: float = 1.5, vmax: float = 4.7):
    kphi = np.zeros(speeds.shape)
    kphidots = np.zeros(speeds.shape)
    for i, speed in enumerate(speeds):
        # if speed < vmin:
        #     kphidots[i] = -kv * ((vmax - vmin) / vmin) * speed
        if speed <= vmax:
            kphidots[i] = -kv * (vmax - speed)
        elif speed > vmax:
            kphi[i] = kc * (speed - vmax)

    return speeds, kphi, kphidots


if __name__ == "__main__":
    main()
