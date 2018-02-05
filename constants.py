# batch size
BATCH_SIZE=5
EPOCHS=10

# data directories
TRAINING_DIR="./data/train/"
FEATURE_DIR=TRAINING_DIR + "images/"
LABEL_DIR=TRAINING_DIR + "labels/"

TESTING_DIR="./data/test/"



# model directory
MODEL_DIR="./model/"

# tensorboard directory
TENSORBOARD_DIR="./tensorboard/"

# dataset size 
TRAINING_DATASET_SIZE=15000
TESTING_DATASET_SIZE=2000

NUM_BATCHES=TRAINING_DATASET_SIZE/BATCH_SIZE
