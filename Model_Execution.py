import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint

# prepare usefull callbacks
lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=7, min_lr=10e-7, epsilon=0.01, verbose=1)
early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1)
checkpoint = ModelCheckpoint('/content/drive/MyDrive/CASIA 2.0/Temp/CasiaAttention_BestModel4.h5', monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='auto')
callbacks= [lr_reducer, early_stopper,checkpoint]

# define loss, metrics, optimizer
model.compile(keras.optimizers.Adam(lr=0.0004), loss='categorical_crossentropy', metrics=['accuracy'])

# fits the model on batches with real-time data augmentation
batch_size = 64
model.fit(X_train, y_train, batch_size=batch_size,
          steps_per_epoch=len(X_train)//batch_size, epochs=100,
          validation_data=(X_test, y_test), 
          validation_steps=len(X_test)//batch_size,
          callbacks=callbacks)