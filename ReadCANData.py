import os
import can_decoder
import mdf_iter
import argparse

from datetime import timedelta


def main():
    parser = argparse.ArgumentParser(
        description="Read MDF files from input directory, convert to CSV and save to output directory."
    )
    parser.add_argument('input_directory', type=str, help="Directory that contains MDF files")
    parser.add_argument('output_directory', type=str, help="Directory in which CSV files will be saved")
    args = parser.parse_args()

    dbc_path = os.path.join("dbc_files", "Fused_new.dbc")
    db = can_decoder.load_dbc(dbc_path, use_custom_attribute="SPN")
    df_decoder = can_decoder.DataFrameDecoder(db)


    for filename in os.listdir(args.input_directory):
        path = os.path.join(args.input_directory, filename)
        print(f"Decoding {path}...")
        
        with open(path, "rb") as file_handle:
            mdf_file = mdf_iter.MdfFile(file_handle)
            df_raw = mdf_file.get_data_frame()

        df_finalized = df_decoder.decode_frame(df_raw)
        df_finalized.index += timedelta(hours=2)  # Add 2 hours to compensate for time difference

        timestamp = df_finalized.index[0]
        output_name = f"{timestamp.year}-{timestamp.month}-{timestamp.day}_{timestamp.hour}-{timestamp.minute}.csv"
        output_path = os.path.join(args.output_directory, output_name)

        if os.path.isfile(output_path):
            print(f"File with name {output_path} already exists, skipping...")
            continue

        df_finalized.to_csv(output_path)
        print(f"Saved to {output_path}.")


if __name__ == "__main__":
    main()