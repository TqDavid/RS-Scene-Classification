from keras.metrics import top_k_categorical_accuracy


# 全局变量
PATH = 'F:/GRADUATION/experimental_dataset/场景分类数据集/UCMerced_LandUse/Images/'
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3
NUM_CLASSES = 21
BATCH_SIZE = 20
EPOCHS = 100
OPTIMIZER = 'Adam'
LOSS = 'categorical_crossentropy'
MERTICS = ['acc', top_k_categorical_accuracy]
GPUS = '0'
TEST_SIZE = 0.25