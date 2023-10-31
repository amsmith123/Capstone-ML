import tensorflow as tf
from keras import layers
import os

# Define a simple CNN for binary classification
def create_classification_model(input_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load the training and testing data
def load_images(folder_path):
    image_list = []
    gt_list = []
    
    # List all files in the folder
    files = os.listdir(folder_path)
    # files are already sorted; if not, use file.sort()
    
    for filename in files:
        if filename.endswith('.png'):
            # Full path to the file
            full_path = os.path.join(folder_path, filename)
            
            # Read the image
            img = cv2.imread(full_path)
            
            # Check if it's a GT image or a regular image
            if '_GT' in filename:
                gt_list.append(img)
            else:
                image_list.append(img)
                
    return image_list, gt_list


    
train_images, train_gt = load_images('Data/train')
test_images, test_gt = load_images('Data/test')

# Load a split file
split_info = load_pyb('Data/split_weakly_0.pyb')

# Prepare the data based on the split_info
train_dataset = prepare_dataset(train_images, train_gt, split_info['train'])
val_dataset = prepare_dataset(train_images, train_gt, split_info['val'])

# Define and compile model
model = create_classification_model((230, 630, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print(model.summary())

# Train the model
# model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)
    





