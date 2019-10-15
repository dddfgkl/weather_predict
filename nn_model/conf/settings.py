from datetime import datetime

DATA_PATH = '../pre_data'

MILESTONES = [100, 130, 160]

#weights file directory
CHECKPOINT_PATH = 'checkpoints'

TIME_NOW = datetime.now().isoformat()

#tensorboard log file directory
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

#input image size for network
#IMAGE_SIZE = 224
IMAGE_H,IMAGE_W = 269, 239
