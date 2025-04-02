```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

    2025-03-26 06:04:58.312476: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2025-03-26 06:04:58.328918: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2025-03-26 06:04:58.347732: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2025-03-26 06:04:58.353285: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2025-03-26 06:04:58.368177: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2025-03-26 06:04:59.223690: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
import pandas as pd
```


```python
data = pd.read_csv('customer_churn_dataset/customer_churn.csv')
```


```python
data.head()
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
      <th>CustomerID</th>
      <th>Tenure</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Contract</th>
      <th>PaymentMethod</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>5</td>
      <td>70.0</td>
      <td>350.0</td>
      <td>Month-to-month</td>
      <td>Electronic check</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>10</td>
      <td>85.5</td>
      <td>850.5</td>
      <td>Two year</td>
      <td>Mailed check</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>3</td>
      <td>55.3</td>
      <td>165.9</td>
      <td>One year</td>
      <td>Electronic check</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>8</td>
      <td>90.0</td>
      <td>720.0</td>
      <td>Month-to-month</td>
      <td>Credit card</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>2</td>
      <td>65.2</td>
      <td>130.4</td>
      <td>One year</td>
      <td>Electronic check</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 7 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   CustomerID      5 non-null      int64  
     1   Tenure          5 non-null      int64  
     2   MonthlyCharges  5 non-null      float64
     3   TotalCharges    5 non-null      float64
     4   Contract        5 non-null      object 
     5   PaymentMethod   5 non-null      object 
     6   Churn           5 non-null      int64  
    dtypes: float64(2), int64(3), object(2)
    memory usage: 408.0+ bytes



```python
data = data.dropna() # Drop missing valeus
```


```python
data.head()
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
      <th>CustomerID</th>
      <th>Tenure</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Contract</th>
      <th>PaymentMethod</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>5</td>
      <td>70.0</td>
      <td>350.0</td>
      <td>Month-to-month</td>
      <td>Electronic check</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>10</td>
      <td>85.5</td>
      <td>850.5</td>
      <td>Two year</td>
      <td>Mailed check</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>3</td>
      <td>55.3</td>
      <td>165.9</td>
      <td>One year</td>
      <td>Electronic check</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>8</td>
      <td>90.0</td>
      <td>720.0</td>
      <td>Month-to-month</td>
      <td>Credit card</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005</td>
      <td>2</td>
      <td>65.2</td>
      <td>130.4</td>
      <td>One year</td>
      <td>Electronic check</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.get_dummies(data, drop_first=True)
```


```python
from sklearn.model_selection import train_test_split
```


```python
X = data.drop('Churn', axis=1)
y = data['Churn']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
]) # Develop model TensorFlow
```

    /anaconda/envs/azureml_py38/lib/python3.10/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)
    2025-03-26 06:05:14.417271: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected



```python
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = nn.functional.dropout(x, 0.5, training=self.training)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = ChurnModel()  # Develop model Pytorch
```


```python
model = RandomForestClassifier(n_estimators=100, random_state=42) # Develop model SciKit-learn
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```


```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

    Epoch 1/10


    /anaconda/envs/azureml_py38/lib/python3.10/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1s/step - accuracy: 0.2500 - loss: 140.6521 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 2/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - accuracy: 0.2500 - loss: 131.2956 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 3/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - accuracy: 0.2500 - loss: 121.9464 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 4/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - accuracy: 0.2500 - loss: 112.5928 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 5/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - accuracy: 0.2500 - loss: 103.2494 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 6/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - accuracy: 0.2500 - loss: 93.9157 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 7/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - accuracy: 0.2500 - loss: 84.6300 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 8/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - accuracy: 0.2500 - loss: 75.3538 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 9/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - accuracy: 0.2500 - loss: 66.0863 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
    Epoch 10/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - accuracy: 0.2500 - loss: 56.8586 - val_accuracy: 1.0000 - val_loss: 0.0000e+00





    <keras.src.callbacks.history.History at 0x7feb8bb16fb0>




```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}') # Model evaluation
```

    1/1 - 0s - 17ms/step - accuracy: 1.0000 - loss: 0.0000e+00
    Test accuracy: 1.0



```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert() # Model optimization
```

    INFO:tensorflow:Assets written to: /tmp/tmp1h_woidi/assets


    INFO:tensorflow:Assets written to: /tmp/tmp1h_woidi/assets


    Saved artifact at '/tmp/tmp1h_woidi'. The following endpoints are available:
    
    * Endpoint 'serve'
      args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 8), dtype=tf.float32, name='keras_tensor_5')
    Output Type:
      TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
    Captures:
      140649640558656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140649637635600: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140649637635248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      140649637631200: TensorSpec(shape=(), dtype=tf.resource, name=None)


    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    W0000 00:00:1742971040.422583    4770 tf_tfl_flatbuffer_helpers.cc:392] Ignored output_format.
    W0000 00:00:1742971040.422645    4770 tf_tfl_flatbuffer_helpers.cc:395] Ignored drop_control_dependency.
    2025-03-26 06:37:20.423176: I tensorflow/cc/saved_model/reader.cc:83] Reading SavedModel from: /tmp/tmp1h_woidi
    2025-03-26 06:37:20.423811: I tensorflow/cc/saved_model/reader.cc:52] Reading meta graph with tags { serve }
    2025-03-26 06:37:20.423823: I tensorflow/cc/saved_model/reader.cc:147] Reading SavedModel debug info (if present) from: /tmp/tmp1h_woidi
    2025-03-26 06:37:20.427472: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:388] MLIR V1 optimization pass is not enabled
    2025-03-26 06:37:20.428148: I tensorflow/cc/saved_model/loader.cc:236] Restoring SavedModel bundle.
    2025-03-26 06:37:20.450014: I tensorflow/cc/saved_model/loader.cc:220] Running initialization op on SavedModel bundle at path: /tmp/tmp1h_woidi
    2025-03-26 06:37:20.455601: I tensorflow/cc/saved_model/loader.cc:462] SavedModel load for tags { serve }; Status: success: OK. Took 32429 microseconds.
    2025-03-26 06:37:20.655373: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.



```python
model.save('churn_model.h5') # Saving the model
```

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 



```python
model.save('churn_model.keras')
```


```python
model.fit(X_train, y_train)
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - accuracy: 0.2500 - loss: 47.6361





    <keras.src.callbacks.history.History at 0x7feb88f27dc0>




```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Test accuracy: {accuracy}')
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step
    Test accuracy: 1.0



```python
# Simplify model by limiting its maximum depth
pruned_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, max_features='sqrt') 

pruned_model.fit(X_train, y_train) 
pruned_predictions = pruned_model.predict(X_test) 
pruned_accuracy = accuracy_score(y_test, pruned_predictions) 
print(f'Pruned Test accuracy: {pruned_accuracy}')
```

    Pruned Test accuracy: 1.0



```python
import joblib

joblib.dump(model, 'churn_model.pkl')
```




    ['churn_model.pkl']




```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```


```python
# Ensure data is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
# Example model (ensure this is defined before training)
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Output 1 neuron for binary classification
        self.sigmoid = nn.Sigmoid()
```


```python
def forward(self, x):
        return self.sigmoid(self.fc(x))

# Initialize model
input_dim = X_train.shape[1]  # Get number of features
model = BinaryClassifier(input_dim).to(device)
```


```python
# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors and move to the same device as model
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)  # Make sure it's (batch_size, 1)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```


```python
X_batch, y_batch = X_batch.to(device), y_batch.to(device)
```


```python
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
```


```python

```
