import pathlib
from os.path import join as pjoin


def load_signal_from_matfile():
    # Get the path to the .mat file
    # This uses pathlib to automatically get the folder where this script is located and navigates from there
    script_folder = pathlib.Path(__file__).parent.resolve()
    data_dir = pjoin(script_folder, "../", "data")
    mat_fname = pjoin(data_dir, "lab_e.mat")

    # Load data
    # Add code here


if __name__ == "__main__":
    t, v = load_signal_from_matfile()
    # Add code here
