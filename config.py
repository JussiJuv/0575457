#Data configuration
ROOT_DIR = './training'          # Main training data
VAL_DIR = './validation'         # Validation data
TEST_DIR = './testing'           # !!! UPDATE THIS LINE !!!
SKELETON_ROOT = './training_skeleton'
VAL_SKELETON_ROOT = './validation_skeleton'
TEST_SKELETON_ROOT = './testing_skeleton'   # !!! UPDATE THIS LINE !!!
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 1

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4

NUM_FRAMES = 4

# Model configuration
NUM_CLASSES = 32
CHECKPOINT_DIR = "checkpoints"