import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
file=pd.read_csv("data.txt")
bact_count=list(file["bacterial_count"])
scc=list(file["somatic_cell_count"])
data=[]
for i in range(0,2000,1):
  data.append([bact_count[i],scc[i]])
print(data)
labels=list(file["status"])
print(labels)
labels=np.array(labels)
data=np.array(data)
print(data.shape)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2)
print(train_data.shape)
print(train_data[0].shape)
# Building the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=train_data[0].shape),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax'),
])

# Compiling the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fitting/Training the model
model.fit(train_data, train_labels, epochs=250)

bacterial_input=float(input("Enter a Bacterial Count (# CFUs per 1 mL) @ 30 Degrees Celsius:"))
scc_input=float(input("Enter a Somatic Cell Count for every 1 mL:"))
input_data=[bacterial_input,scc_input]
input_data=np.array(input_data)
input_data_reshaped=np.reshape(input_data,(1,2,))
prediction=model.predict(input_data_reshaped)
prediction=prediction[0]
max_index=np.argmax(prediction)
print(prediction)
if max_index==0:
  print("Safe for Consumption")
else:
  print("Not Safe for Consumption")
