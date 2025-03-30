# reasoning-scheduling

`gen_datasets.py` - pulls data from huggingface and saves it in the Dyanor data directory. Run once and never again for each of the datasets we want to add.

`mlp_datagen.py` - generates X,y for our prediction task. E.g. y is the early stopping probabilities of generating the correct answer if we terminate after W, 2W, .... tokens. X is the predictors we want to use, e.g. the hidden states and potentially other data. This script generates X and y for given datasets and saves it to netscratch.

`generate_data.sh` - sbatch script that batches out the data generation jobs with mlp_datagen.py`

`mlp_train.py` loads in data, trains in MLP, and saves the MLP

`mlp.py` - defines the MLP class

`Dynasor/benchmark/TokenDeprivation/run_adaptive.py` - script that loads in MLP predictions and/or oracle data and records the tokens per accuracy given adaptive greedy strategy of allocating more size/reasoning budget to queries that will benefit from it more

`run.sh` - runs benchmark tokens vs accuracy experiment for non-adaptive allocation strategy

`run_adaptive.sh` - runs benchmark tokens vs accuracy experiment for adaptive allocation strategy using MLP predictions

`run_adaptive_oracle.sh` - runs benchmark tokens vs accuracy experiment for adaptive allocation strategy using oracle ground truth for early stopping correctness probabilities (using the same data used to train the MLP)

`vis_adaptive.py` - creates visualization for tokens vs accuracy experiment for non-adaptive vs adaptive vs oracle