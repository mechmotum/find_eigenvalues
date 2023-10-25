import os
import can_decoder
import mdf_iter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

DBC_PATH = os.path.join("dbc_files", "Fused_new.dbc")


def main():
    seg = pd.read_excel("segmentation.xlsx")
    mf4 = seg["file"].values[0]
    bikedata = decode_MF4(
        os.path.join("eigenvalues_balance_assist_120523", mf4), DBC_PATH
    )
    print(bikedata["Signal"].unique())

    start_time = bikedata.loc[bikedata["Signal"] == "gyro_x"].index[0]
    speed = get_signal(bikedata, "ws_rear", start_time)
    roll_angle = get_signal(bikedata, "roll", start_time)
    roll_rate = get_signal(bikedata, "gyro_x", start_time)
    steer_rate = get_signal(bikedata, "LWS_SPEED", start_time)
    current_log = get_signal(bikedata, "iq_current_SET", start_time)
    current_calc = calculate_controller_current(
        speed, roll_rate, roll_angle, steer_rate
    )

    signals = [
        {"name": "speed", "df": speed, "units": "m/s", "column": "Physical Value"},
        {
            "name": "roll_angle",
            "df": roll_angle,
            "units": "rad",
            "column": "Physical Value",
        },
        {
            "name": "roll_rate",
            "df": roll_rate,
            "units": "rad/s",
            "column": "Physical Value",
        },
        {
            "name": "steer_rate",
            "df": steer_rate,
            "units": "deg/s",
            "column": "Physical Value",
        },
        {
            "name": "current_log",
            "df": current_log,
            "units": "A",
            "column": "Physical Value",
        },
        {"name": "current_calc", "df": current_calc, "units": "A", "column": "current"},
    ]
    plot(signals)


def plot(signals) -> None:
    """Plots the signals in the signals dictionary."""
    _, axs = plt.subplots(len(signals), 1, sharex=True)

    for i, s in enumerate(signals):
        s["df"].plot(
            x="seconds_since_start",
            y=s["column"],
            ax=axs[i],
            grid=True,
            label=s["name"],
        )
        axs[i].set_ylabel(s["name"] + "[" + s["units"] + "]")

    axs[-1].set_xlabel("Time since start [s]")
    plt.show()


def decode_MF4(path_to_MF4: str, dbc_path: str) -> pd.DataFrame:
    """Decodes .MF4 file with DBC file and returns it in pandas DataFrame format.

    Parameters
    ----------
    path_to_MF4 : str
        Path to the MF4 file that should be decoded.
    path_to_DBC : str
        Path to the DBC file that should be used for decoding.

    Returns
    -------
    df_decoded : pandas.DataFrame
        The decoded MF4 file as a DataFrame.
    """
    db = can_decoder.load_dbc(dbc_path)
    df_decoder = can_decoder.DataFrameDecoder(db)

    with open(path_to_MF4, "rb") as handle:
        mdf_file = mdf_iter.MdfFile(handle)
        df_raw = mdf_file.get_data_frame()

    df_decoded = df_decoder.decode_frame(df_raw)
    df_decoded.index = df_decoded.index.tz_convert("Europe/Amsterdam")
    return df_decoded


def get_signal(
    df: pd.DataFrame, signal_name: str, start_time: datetime
) -> pd.DataFrame:
    """Returns a single signal from the larger dataframe.

    Parameters
    ----------
    df : pandas.DataFrame:
        Dataframe that contains the signal.
    signal_name : str
        Name of the signal to extract from dataframe.

    Returns
    -------
    signal_added_time : pandas.DataFrame
        DataFrame containing only the selected signal.
    """
    try:
        signal = df.loc[df["Signal"] == signal_name]
    except Exception as e:
        print(
            f"Signal with name {signal_name} gave the following exception: {e}. Skipping..."
        )
        return

    signal_added_time = signal.assign(
        seconds_since_start=(signal.index - start_time).total_seconds()
    )

    # if signal_name in ["gyro_x", "roll"]:
    #     signal_added_time.loc[:, "Physical Value"] = (
    #         signal_added_time["Physical Value"] / (2 * np.pi) * 360
    #     )

    return signal_added_time.drop(["CAN ID", "Signal"], axis=1)


def calculate_controller_current(
    speed, roll_rate, roll_angle, steer_rate
) -> pd.DataFrame:
    """Calculates the current that the controller should theoretically send.
    Implements the same controller as on the Teensy that is in the bicycle.

    TODO: implement notch and lowpass filter such as in bicycle.

    Returns
    -------
    combinded_data : pd.DataFrame
        The same dataframe, but with an added column representing controller current.
    """
    vmax = 4.7
    vmin = 1.5
    kc = -0.7
    kv = -6
    cd = 0

    controller = pd.DataFrame()
    controller["seconds_since_start"] = speed["seconds_since_start"]
    controller["current"] = np.nan

    for df in [speed, roll_rate, roll_angle, controller]:
        df.drop(df.tail(2).index, inplace=True)
        df.reset_index(inplace=True)
    steer_rate.reset_index(inplace=True)

    smaller_than_min = speed.index[speed["Physical Value"] < vmin].tolist()
    between_min_max = speed.index[
        (speed["Physical Value"] > vmin) & (speed["Physical Value"] < vmax)
    ].tolist()
    bigger_max = speed.index[speed["Physical Value"] > vmax].to_list()

    print(controller)

    controller.loc[smaller_than_min, "current"] = (
        -kv
        * ((vmax - vmin) / vmin * speed.loc[smaller_than_min, "Physical Value"])
        * roll_rate.loc[smaller_than_min, "Physical Value"]
        + cd * steer_rate.loc[smaller_than_min, "Physical Value"]
    )
    controller.loc[between_min_max, "current"] = (
        -kv
        * (vmax - speed.loc[between_min_max, "Physical Value"])
        * roll_rate.loc[between_min_max, "Physical Value"]
        + cd * steer_rate.loc[between_min_max, "Physical Value"]
    )
    controller.loc[bigger_max, "current"] = (
        kc
        * (speed.loc[bigger_max, "Physical Value"] - vmax)
        * roll_angle.loc[bigger_max, "Physical Value"]
        + cd * steer_rate.loc[bigger_max, "Physical Value"]
    )

    controller.loc[controller["current"] < -7, "current"] = -7
    controller.loc[controller["current"] > 7, "current"] = 7
    controller.loc[:, "current"] *= 5

    return controller


if __name__ == "__main__":
    main()
