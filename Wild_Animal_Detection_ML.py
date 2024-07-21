#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import sys
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import legacy 


# In[2]:


train_dir=r"C:\Users\Akash Baskar\Desktop\animal_detection"
val_dir=r"C:\Users\Akash Baskar\Desktop\animal_detection_train"


# In[3]:


class_labels=os.listdir(train_dir)
print(class_labels)
IMAGE_SIZE=30


# In[7]:


def create_training_data():
    training_date = []
    for categories in class_labels:
        path = os.path.join(train_dir,categories)
        class_num = class_labels.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMAGE_SIZE,IMAGE_SIZE))
                training_date.append([new_array,class_num])
            except:
                pass
    return training_dat


# In[5]:


train_total=0
for label in class_labels:
    total=len(os.listdir(os.path.join(train_dir,label)))
    print(label,total)
    train_total+=total
print('Total....',train_total)


# In[6]:


val_total=0
for label in class_labels:
    total=len(os.listdir(os.path.join(val_dir,label)))
    print(label,total)
    val_total+=total
print('Total....',val_total)


# In[8]:


nb_train_samples=train_total
nb_val_samples=val_total
num_classes=5
img_rows=128
img_cols=128
channel=3


# In[11]:


x_train = []
y_train = []

for j, label in enumerate(class_labels):
    image_names_train = os.listdir(os.path.join(train_dir, label))
    total = len(image_names_train)
    print(label, total)
    for image_name in image_names_train:
        try:
            img = image.load_img(os.path.join(train_dir, label, image_name), target_size=(img_rows, img_cols))
            img = image.img_to_array(img)
            img = img / 255.0
            x_train.append(img)
            y_train.append(j)
        except:
            pass

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = np_utils.to_categorical(y_train[:nb_train_samples], num_classes)


# In[12]:


x_test = []
y_test = []

for j, label in enumerate(class_labels):
    image_names_test = os.listdir(os.path.join(val_dir, label))
    total = len(image_names_test)
    print(label, total)
    for image_name in image_names_test:
        try:
            img = image.load_img(os.path.join(val_dir, label, image_name), target_size=(img_rows, img_cols))
            img = image.img_to_array(img)
            img = img / 255.0
            x_test.append(img)
            y_test.append(j)
        except:
            pass

x_test = np.array(x_test)
y_test = np.array(y_test)
y_test = np_utils.to_categorical(y_test[:nb_val_samples], num_classes)


# In[13]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[16]:


plt.imshow(x_test[10])
plt.show()


# In[15]:


model=Sequential()


# In[17]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()


# In[18]:


# Training the model
history = model.fit(
    x_train, y_train,
    epochs=10,  # Adjust the number of epochs as needed
    batch_size=32,  # Adjust the batch size as needed
    validation_data=(x_test, y_test)
)


# In[19]:


# Evaluating the model
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# In[20]:


# Plotting the training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()


# In[21]:


# Saving the model
model.save('wild_animal_detection_model.h5')


# In[33]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('wild_animal_detection_model.h5')

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128, 3))  # Use the target size used in training
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    return img

# Define class labels (make sure this matches the order used during training)
class_labels = ['elephant', 'gaur', 'leopard', 'lion', 'tiger']

# Path to the new image (using raw string for Windows file path)
new_image_path = r'C:\Users\Akash Baskar\Desktop\animal_detection_train\Elephant\as_te36.jpg'

# Preprocess the image
new_img = preprocess_image(new_image_path)

# Make a prediction
prediction = model.predict(new_img)
predicted_class = np.argmax(prediction, axis=1)

# Map the predicted class index to the class label
predicted_class_label = class_labels[predicted_class[0]]

# Display the image and prediction
original_img = image.load_img(new_image_path)
plt.imshow(original_img)
plt.title(f'Predicted class: {predicted_class_label}')
plt.axis('off')  # Hide axes for better visualization
plt.show()


# In[ ]:




