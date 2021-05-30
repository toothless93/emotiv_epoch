import os


def create_folders(log_dir):
    for i in log_dir:
        if not os.path.exists(i):
            os.mkdir(i)
