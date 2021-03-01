import datetime
import os
import pathlib

# Put the path to your UCR archive here (best used absolute path to avoid surprises)
# On linux, something like:
# UCR_ARCHIVE_PATH = "/home/user/Univariate_ts"

# !!! HERE !!!
UCR_ARCHIVE_PATH = ""

# Alternatively, you can set an environment variable named "UCR_ARCHIVE_PATH"
# On linux, in your shell configuration file (.bashrc, .zshrc,...) or .profile

# --- --- --- Tooling

def get_ucr_folder():
    global UCR_ARCHIVE_PATH
    # If empty, check the environment variable
    if UCR_ARCHIVE_PATH == "":
        try:
            UCR_ARCHIVE_PATH = os.environ["UCR_ARCHIVE_PATH"]
        except KeyError:
            print("It looks like 'UCR_ARCHIVE_PATH' is not set")
            print(f"Have a look at {pathlib.Path(__file__).absolute()}")
            exit(1)
    # We should have a path now. Check it
    folder = pathlib.Path(UCR_ARCHIVE_PATH).absolute()
    if not (folder.exists() and folder.is_dir()):
        print("I could not find your UCR archive folder:")
        print(" -->  " + str(folder))
        exit(1)
    return folder


# Tooling for our scripts
def get_timestemp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")

