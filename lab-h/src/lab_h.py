import pathlib
from os.path import join as pjoin


def read_email_file():
    # Get the path to the .txt file
    # This uses pathlib to automatically get the folder where this script is located and navigates from there
    script_folder = pathlib.Path(__file__).parent.resolve()
    data_dir = pjoin(script_folder, "../", "data")
    fname = pjoin(data_dir, "email.txt")

    # Load data
    # Add code here


if __name__ == "__main__":
    # Add code here
