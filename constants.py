import os

# Get the user from environment variable
USER = os.environ.get('USER', '')

# Define base paths based on user
#if USER == "katrinabrown":
if True:
    # Paths for katrinabrown
    BASE_PATH = "/n/netscratch/dwork_lab/Lab/katrina"
    STEM = f"{BASE_PATH}/reasoning_scheduling_new/"
    X_STEM = f"{STEM}/data/"
    Y_STEM = f"{STEM}/data/"
else:
    # Paths for amuppidi
    BASE_PATH = "/n/netscratch/gershman_lab/Lab/amuppidi"
    STEM = f"{BASE_PATH}/reasoning_scheduling_new/"
    X_STEM = f"{BASE_PATH}/reasoning_scheduling_new_orig/data/"
    Y_STEM = f"{STEM}/data/"

