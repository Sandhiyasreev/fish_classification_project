from tensorflow.keras.applications import MobileNet, VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from data_loader import load_data

def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(train_gen.num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

train_gen, val_gen = load_data(
    "data/images.cv_jzk6llhf18tm3k0kyttxz/data/train",
    "data/images.cv_jzk6llhf18tm3k0kyttxz/data/val"
)


for model_name, model_class in [('mobilenet', MobileNet), ('vgg16', VGG16), ('resnet50', ResNet50)]:
    base = model_class(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False
    model = build_model(base)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(f'models/{model_name}_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    model.fit(train_gen, epochs=5, validation_data=val_gen, callbacks=[checkpoint])
