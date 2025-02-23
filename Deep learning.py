# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:45:09 2025

@author: aliel
"""


import pandas as pd 
import numpy as np 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input , Add, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam



test_df = pd.read_csv(r'C:\Users\aliel\Downloads\digit-recognizer\test.csv')
train_df = pd.read_csv(r'C:\Users\aliel\Downloads\digit-recognizer\train.csv')

train_df.shape
test_df['label'] = -1


merged_df = pd.concat([test_df, train_df], ignore_index=True)

#PreProcessing 

test_df = merged_df[merged_df['label'] == -1]
train_df = merged_df[merged_df['label'] != -1]

y_train = train_df['label'].values
x_train = train_df.drop(columns=['label']).values
x_test = test_df.drop(columns=['label']).values

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)


x_train_small, x_val, y_train_small, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
 
#CNN MODEL
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
    
    
    ])


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#training model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_val, y_val)
print(f'CNN accuracy : {test_acc}')



# MLP model
mlp_model = Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Flatten image data into a 1D array
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Output layer for 10 classes
])

mlp_model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
#train mlp
mlp_history = mlp_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

# Evaluate the MLP model
mlp_loss, mlp_acc = mlp_model.evaluate(x_val, y_val)
print(f"MLP Accuracy: {mlp_acc}")


#ResNet Model
def build_resnet(input_shape=(28,28,1), num_classes=10):
    inputs = Input(shape=input_shape)
    
    x= Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x= BatchNormalization()(x)
    
    shortcut = x 
    x= Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x= BatchNormalization()(x)
    x= Conv2D(64, (3,3), padding='same')(x)
    x= BatchNormalization()(x)
    x= Add()([x, shortcut])
    x= ReLU()(x)
    
    x = MaxPooling2D(2,2)(x)
    
    shortcut = Conv2D(128, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    x= Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x= BatchNormalization()(x)
    x= Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x= BatchNormalization()(x)
    x= Conv2D(128, (3, 3), padding='same')(x)
    x= BatchNormalization()(x)
    x= Add()([x, shortcut])
    x= ReLU()(x)
    
    x= Flatten()(x)
    x= Dense(256, activation='relu')(x)
    x= BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model_1 = Model(inputs, outputs)
    return model_1

#create & compile the model
resnet_model = build_resnet()
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
history_resnet = resnet_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

#evaluate the model 
resnet_loss, resnet_acc = resnet_model.evaluate(x_val, y_val)
print(f"ResNet Accuracy: {resnet_acc}")


# Predict on the test set
y_pred = mlp_model.predict(x_test)

# Convert predictions from one-hot encoding back to labels
y_pred_labels = y_pred.argmax(axis=1)

# Prepare the submission file
submission = pd.DataFrame({'ImageId': range(1, len(y_pred_labels) + 1), 'Label': y_pred_labels})
submission.to_csv('submission2.csv', index=False)

print("Submission file created: submission.csv")






    
    



