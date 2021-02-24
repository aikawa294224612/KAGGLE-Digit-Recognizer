# KAGGLE-Digit-Recognizer
kaggle: https://www.kaggle.com/c/digit-recognizer/overview

**Public Score**: 0.98567

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/digit-recognizer/sample_submission.csv
    /kaggle/input/digit-recognizer/train.csv
    /kaggle/input/digit-recognizer/test.csv
    


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras import backend as K
```


```python
training_path = "/kaggle/input/digit-recognizer/train.csv"
testing_path = "/kaggle/input/digit-recognizer/test.csv"

epoch = 15
batchsize = 64
image_size = 28
```


```python
data = pd.read_csv(training_path)
data.head()

data_test = pd.read_csv(testing_path)
data_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>




```python
y_train = data.loc[:, "label"]
x_train = data.iloc[:, 1:]

x_test = data_test.iloc[:, :]

print("Training set:", x_train.shape)
print("Training set labels:", y_train.shape)

print("Testing set:", x_test.shape)
```

    Training set: (42000, 784)
    Training set labels: (42000,)
    Testing set: (28000, 784)
    


```python
x_train = x_train.values.reshape(-1, 1, image_size, image_size)
x_test = x_test.values.reshape(-1, 1, image_size, image_size)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
```


```python
y_train = keras.utils.to_categorical(y_train)

print(y_train[0])
```

    [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
    


```python
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, image_size, image_size), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()
```

    Model: "sequential_9"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_10 (Conv2D)           (None, 1, 28, 32)         8096      
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 1, 28, 32)         128       
    _________________________________________________________________
    activation_8 (Activation)    (None, 1, 28, 32)         0         
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 1, 14, 16)         0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 1, 14, 16)         0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 1, 14, 64)         9280      
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 1, 14, 64)         256       
    _________________________________________________________________
    activation_9 (Activation)    (None, 1, 14, 64)         0         
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 1, 7, 32)          0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 1, 7, 32)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 224)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 512)               115200    
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 512)               2048      
    _________________________________________________________________
    activation_10 (Activation)   (None, 512)               0         
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 140,138
    Trainable params: 138,922
    Non-trainable params: 1,216
    _________________________________________________________________
    


```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
h = model.fit(x_train, y_train, epochs=epoch, batch_size=batchsize, verbose=1, validation_split=0.15)

print(h.history.keys())
print("Acc:", h.history['accuracy'][-1])
print("Val Acc:", h.history['val_accuracy'][-1])

def show_train_history(train_history, train, validation, title):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title(title)  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show() 

show_train_history(h, 'accuracy', 'val_accuracy', 'Train History') 
show_train_history(h, 'loss', 'val_loss', 'Loss History')  
```

    Epoch 1/15
    558/558 [==============================] - 3s 4ms/step - loss: 0.7141 - accuracy: 0.7622 - val_loss: 0.1103 - val_accuracy: 0.9638
    Epoch 2/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.2018 - accuracy: 0.9365 - val_loss: 0.0859 - val_accuracy: 0.9732
    Epoch 3/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.1599 - accuracy: 0.9487 - val_loss: 0.0731 - val_accuracy: 0.9754
    Epoch 4/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.1291 - accuracy: 0.9586 - val_loss: 0.0757 - val_accuracy: 0.9746
    Epoch 5/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.1257 - accuracy: 0.9601 - val_loss: 0.0639 - val_accuracy: 0.9771
    Epoch 6/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.1099 - accuracy: 0.9639 - val_loss: 0.0546 - val_accuracy: 0.9827
    Epoch 7/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.0996 - accuracy: 0.9682 - val_loss: 0.0515 - val_accuracy: 0.9840
    Epoch 8/15
    558/558 [==============================] - 3s 5ms/step - loss: 0.0959 - accuracy: 0.9689 - val_loss: 0.0524 - val_accuracy: 0.9817
    Epoch 9/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.0906 - accuracy: 0.9702 - val_loss: 0.0537 - val_accuracy: 0.9822
    Epoch 10/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.0828 - accuracy: 0.9724 - val_loss: 0.0501 - val_accuracy: 0.9843
    Epoch 11/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.0826 - accuracy: 0.9731 - val_loss: 0.0595 - val_accuracy: 0.9827
    Epoch 12/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.0796 - accuracy: 0.9743 - val_loss: 0.0472 - val_accuracy: 0.9856
    Epoch 13/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.0705 - accuracy: 0.9765 - val_loss: 0.0494 - val_accuracy: 0.9841
    Epoch 14/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.0718 - accuracy: 0.9763 - val_loss: 0.0474 - val_accuracy: 0.9859
    Epoch 15/15
    558/558 [==============================] - 2s 4ms/step - loss: 0.0714 - accuracy: 0.9767 - val_loss: 0.0441 - val_accuracy: 0.9860
    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    Acc: 0.9761624932289124
    Val Acc: 0.9860317707061768
    


![png](output_8_1.png)



![png](output_8_2.png)



```python
predictions = model.predict(x_test)
predictions = np.argmax(predictions,axis=1)

print(predictions[:5])
```

    [2 0 9 0 3]
    


```python
ids = range(1, (len(predictions)+1))

submission = pd.DataFrame({
    "ImageId": ids, 
    "Label": predictions,
})

submission.to_csv("submission.csv", index = False)
```


```python

```
