import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from data import balance_assist_without_rider, balance_assist_with_rider, rigid_bike_without_rider
from model import SteerControlModel

def main():
    parameter_set = Meijaard2007ParameterSet(rigid_bike_without_rider, True)
    model = SteerControlModel(parameter_set)
    fig, ax = plt.subplots()
    speeds = np.linspace(0.0, 6.0, num=61)
    velocities = [6, 8, 10, 12, 14, 16, 18]
    velocities_ms = [vel / 3.6 for vel in velocities]
    colors_measured = ['cornflowerblue', 'blue', 'darkblue']
    colors_theoretical = ['lightgray', 'gray', 'black']
    
    for i, gain in enumerate([6, 8, 10]):
        if gain == 6:
            prefix = "12-May-2023-10-11-04-bas-on-gain-6-speed-"
        else:
            prefix = "12-May-2023-10-39-32-bas-on-gain-" + str(gain) + "-speed-"
        real, img, r_squared = fit_curve(gain, prefix)
    
        kphidots = -gain*(5.0 - speeds)
        kphidots[50:] = 0.0
        ax = model.plot_eigenvalue_parts(ax=ax, v=speeds, kphidot=kphidots, colors = [colors_theoretical[i], colors_theoretical[i], colors_theoretical[i], colors_theoretical[i]])
        print(real)
        plt.plot(
            velocities_ms, real, '.', label="Measured real eigenvalues", color=colors_measured[i]
        )
        plt.plot(
            velocities_ms, img, '.', label='Measured imaginary eigenvalues', color=colors_measured[i]
        )
        # plt.legend()
        print(f"R-squared values for gain {gain}: Mean: {np.mean(r_squared)}, Max: {np.max(r_squared)}, Min: {np.min(r_squared)}")

    fig.show()
    fig.suptitle('Theoretical eigenvalues of rigid bicycle compared to measured eigenvalues at gain of -6, -8 and -10', wrap=True)
    plt.savefig("figures/all-gains", dpi=300)
    fig.waitforbuttonpress()



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
            popt, pcov = spo.curve_fit(kooijman_func,
                                df_gyro.loc[:,"time"],
                                df_gyro.loc[:,"gyro_x"],
                                p0=(-1.0, -1.0, 1.0, 3.0, 1.0))
            eigenvalues_real.append(popt[1])
            eigenvalues_img.append(popt[3])

            # Calculate r-squared
            residuals = df_gyro.loc[:,"gyro_x"] - kooijman_func(df_gyro.loc[:,"time"], *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((df_gyro.loc[:,"gyro_x"] - np.mean(df_gyro.loc[:,"gyro_x"]))**2)
            r_squared.append(1 - (ss_res / ss_tot))


            if show_plots:
                print(popt)
                fig, ax = plt.subplots(1,1)
                ax.plot(df_gyro.loc[:,"time"], df_gyro.loc[:,"gyro_x"], '.')
                ax.plot(
                    df_gyro.loc[:,"time"],
                    kooijman_func(
                        df_gyro.loc[:,"time"], popt[0], popt[1], popt[2], popt[3], popt[4]
                    )
                )
                ax.set_xlabel("Time since perturbation [s]")
                ax.set_ylabel("Roll rate [rad/s]")
                fig.show()
                # fig.waitforbuttonpress()


        avg_eigenvalues_real.append(sum(eigenvalues_real)/np.size(eigenvalues_real))
        avg_eigenvalues_img.append(sum(eigenvalues_img)/np.size(eigenvalues_img))

    return avg_eigenvalues_real, avg_eigenvalues_img, r_squared

    # plot_theoretical_eigenvalues(gain)

    # print(f"Real: {avg_eigenvalues_real}")
    # print(f"Imaginary: {avg_eigenvalues_img}")

    # velocities_ms = [vel / 3.6 for vel in velocities]
    # plt.plot(
    #     velocities_ms, avg_eigenvalues_real, '.', label="Measured real eigenvalues", color='k'
    # )
    # plt.plot(
    #     velocities_ms, avg_eigenvalues_img, '.', label='Measured imaginary eigenvalues', color='c'
    # )
    # # plt.legend()
    # # plt.savefig("figures/gain-" + str(gain) + "-batavus-without-rider", dpi=300)
    # plt.waitforbuttonpress()


def kooijman_func(t, c1, d, c2, omega, c3):
    return c1 + np.exp(d*t) * (c2*np.cos(omega*t) + c3*np.sin(omega*t))


def plot_theoretical_eigenvalues(gain: int):
    parameter_set = Meijaard2007ParameterSet(balance_assist_without_rider, True)
    model = SteerControlModel(parameter_set)

    fig, ax = plt.subplots()
    speeds = np.linspace(0.0, 10.0, num=101)
    kphidots = -gain*(5.0 - speeds)
    kphidots[50:] = 0.0
    ax = model.plot_eigenvalue_parts(ax=ax, v=speeds, kphidot=kphidots, colors = ['k', 'k', 'k', 'k'])
    fig.suptitle('Theoretical eigenvalues of Batavus Browser versus measured eigenvalues at gain of -' + str(gain), wrap=True)
    fig.show()


if __name__ == "__main__":
    main()
