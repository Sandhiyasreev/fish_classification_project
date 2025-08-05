from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, val_dir, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical'
    )
    return train_gen, val_gen
