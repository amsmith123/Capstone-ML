from keras import layers
from keras import Model
import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
from sklearn.metrics import classification_report

# Define absolute directory path
abs_dirpath= os.path.abspath(os.path.dirname(__file__))


# Create U-Net Model
def create_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Output layer: you'll need as many filters as classes you're trying to identify
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs, outputs)
    return model


# Define Dice Loss Function
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1) / (denominator + 1)


def load_images(folder_path):
    image_list = []
    gt_list = []
    filename_to_index = {}  # Create an empty dictionary to hold the mapping
    
    # List all files in the folder
    files = os.listdir(folder_path)
    mapping_index = 0
    for index, filename in enumerate(files):
        if filename.endswith('.png'):
            # Full path to the file
            full_path = os.path.join(folder_path, filename)
            
            # Read the image
            img = cv2.imread(full_path)
            
            # Extract the number from the filename
            number_part = filename.split('.')[0]
            if '_GT' not in number_part:
                try:
                    # Attempt to convert to an integer
                    number = int(number_part)
                    
                    # Add to the mapping
                    filename_to_index[number] = mapping_index
                    mapping_index += 1
                except:
                    pass
                    # print(f"Could not convert {number_part} to an integer.")
                    
            # Check if it's a GT image or a regular image
            if '_GT' in filename:
                gt_list.append(img)
            else:
                image_list.append(img)
            
                
    return image_list, gt_list, filename_to_index


def load_pyb(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return {'train': data[0], 'val': data[1]}


def prepare_dataset(images, gt, split_info, filename_to_index):
    # Create empty lists to hold the selected images and ground truths
    selected_images = []
    selected_gt = []
    
    for idx, label in split_info:
        actual_index = int(filename_to_index.get(idx, None))
        if actual_index is not None:
            #resize images and append to image array
            image = images[actual_index]
            resized_image = cv2.resize(image, (640, 240))
            # resized_image = np.expand_dims(resized_image, axis=0)  # Adds an extra dimension at the beginning

            gt_image = gt[actual_index]
            resized_gt = cv2.resize(gt_image, (640, 240))
            resized_gt = np.expand_dims(resized_gt, axis=0) 
            
            selected_images.append(resized_image)
            selected_gt.append(resized_gt)
        else:
            print(f"ID {idx} not found in filename mapping.")
    
    # Convert the lists to numpy arrays
    selected_images = np.array(selected_images)
    selected_gt = np.array(selected_gt)
    
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((selected_images, selected_gt))
    # dataset = dataset.map(lambda x, y: (x / 255.0, y / 255.0))  # Normalize to [0,1]
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.float32) / 255.0))
    dataset = dataset.batch(32)  # Add batching
    return dataset
    

# Load Images
print("Loading training images")
train_images, train_gt, train_filename_to_index = load_images(f'{abs_dirpath}/Data/train')
print("Loading testing images")
test_images, test_gt, test_filename_to_index = load_images(f'{abs_dirpath}/Data/test')

# Load a split file
split_info = load_pyb(f'{abs_dirpath}/Data/split_weakly_0.pyb')

# Create Datasets
train_dataset = prepare_dataset(train_images, train_gt, split_info['train'], train_filename_to_index)
val_dataset = prepare_dataset(test_images, test_gt, split_info['val'], test_filename_to_index)

# Create and Compile Model
model = create_unet((240, 640, 3))
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# Define the number of epochs
num_epochs = 3

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)

# Save the Model
model.save(f"{abs_dirpath}/saved_models/seg_model_{num_epochs}")