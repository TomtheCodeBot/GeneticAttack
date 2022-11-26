# GeneticAttack.
## _Genetic Attack setup to test Robust Naive Bayes and Multinomial Naive Bayes_

## Setting up environment:

Before running the scripts in this folder, the environment must be installed, using the provided `environment.yml` file. With Anaconda or Miniconda, to create an environemnt, execute this command:

- `conda env create -f environment.yml`

## Building and running Docker container:

For the ease of running the scripts in a docker container, a script was written to simplify the process. In order to run build and run the images, execute the command:

- `./run_docker.sh`

This command will build the docker image and run it on a docker container.

## Running the code manually:

For running the code manually, the steps are as follows:
- Download the neccesary datasets and models:
-- `./setup.sh`
- Create the vector embeddings from  GloVe and calculate a distance matrix.
-- `python build_embeddings.py`
-- `python compute_dist_mat.py`
- Run testing script to output the attack result.
-- `./run_test.sh`

## Config information (Memory error related):

The variables are important for running the test since some of them needs to be adjusted if an out of memory error occured. Inside the config files lies these variables:
- `VOCAB_SIZE` = 25000 (Default:50000), this variable determines the amount of words contained in the dictionary for both the models and the GloVe embeddings. Increase this when a memmory error occurs.
- `SAMPLE_SIZE` = 1000 (Default:1000), this variables sets the amount of samples taken from the testset to test the attack.
- `TEST_SIZE` = 500 (Default:500), the limit to how many cases to run the attack on, if the amount of samples that has already augmented by the attack reaches `TEST_SIZE`, the process will end.
