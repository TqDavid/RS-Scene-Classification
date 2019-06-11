from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
import matplotlib.pyplot as plt


# ---------------------网络结构搭建------------------------- #
# 此处直接调用keras_application中定义好的网络结构，去除全连接层
base_model = ResNet50(weights='imagenet', include_top=False)

# 搭建全连接层，添加Dropout防止过拟合
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(45, activation='softmax', name='predictions')(x)
model = Model(input=base_model.input, output=x)
model.summary()


# 超参数
PATH = '/content/NWPU-RESISC45/'
BATCH_SIZE = 16
TEST_SIZE = 0.3
EPOCHS = 100
CLASSES = 21


# -----------------读取数据（参考kers文档理解以下函数的作用）------------------ #
# 数据扩充
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             rotation_range=10,
                             horizontal_flip=True,
                             validation_split=TEST_SIZE)
# 训练数据
train_generator = datagen.flow_from_directory(PATH, target_size=(256, 256),
                                            classes=None, batch_size=BATCH_SIZE,
                                            subset='training')
# 验证数据
validation_generator = datagen.flow_from_directory(PATH, target_size=(256, 256),
                                            classes=None, batch_size=BATCH_SIZE,
                                            subset='validation')


# -----------------------------------模型训练-------------------------------------- #
# 模型编译
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[top_k_categorical_accuracy, 'acc'])
# 模型断点保存
model_checkpoint = ModelCheckpoint('vgg16.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

# train
result = model.fit_generator(generator=train_generator,
                             steps_per_epoch=train_generator.samples // BATCH_SIZE,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[model_checkpoint],
                             validation_data=validation_generator,
                             validation_steps=validation_generator.samples // BATCH_SIZE,
                             shuffle=True)


# -----------------------------------可视化模型训练曲线------------------------------- #
plt.figure()
plt.plot(result.epoch, result.history['acc'], label="acc")
plt.plot(result.epoch, result.history['val_acc'], label="val_acc")
plt.scatter(result.epoch, result.history['acc'], marker='*')
plt.scatter(result.epoch, result.history['val_acc'])
plt.legend(loc='under right')
plt.show()
plt.figure()
plt.plot(result.epoch, result.history['loss'], label="loss")
plt.plot(result.epoch, result.history['val_loss'], label="val_loss")
plt.scatter(result.epoch, result.history['loss'], marker='*')
plt.scatter(result.epoch, result.history['val_loss'])
plt.legend(loc='upper right')
plt.show()