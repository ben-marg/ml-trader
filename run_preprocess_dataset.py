from compute_indicators_labels_lib import preprocess
from config import RUN as run_conf

if __name__ == '__main__':
    # For frozen executables, you might need freeze_support():
    from multiprocessing import freeze_support
    freeze_support()
    
    # Now run the preprocessing function
    preprocess(run_conf)
