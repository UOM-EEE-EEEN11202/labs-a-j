import pathlib
from os.path import join as pjoin


def load_data():
    # Get the path to the .csv file
    # This uses pathlib to automatically get the folder where this script is located and navigates from there
    script_folder = pathlib.Path(__file__).parent.resolve()
    data_dir = pjoin(script_folder, "../", "data")
    csv_fname = pjoin(data_dir, "lab_f.csv")

    # Load data
    # Add code here


if __name__ == "__main__":
    # Add code here
