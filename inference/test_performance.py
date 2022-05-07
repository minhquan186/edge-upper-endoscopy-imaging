#!/home/mica/.endo_env/bin/python3

# Import libs
import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
import tensorflow as tf
import time
import argparse

# Setup
folder_path="/home/mica/edge-upper-endoscopy-imaging/data_NICS"
class_names = [ '1 Hầu họng', '2 Thực quản', '3 Tâm vị', '4 Thân vị', '5 Phình vị', '6 Hang vị','7 Bờ cong lớn','8 Bờ cong nhỏ','9 Hành tá tràng','10 Tá tràng']
folder_names = ['Vung hau hong', 'Thuc quan', 'Tam vi', 'Than vi','Phinh vi', 'Hang vi','Bo cong lon','Bo cong nho','Hanh ta trang','Ta trang']
nb_classes = len(class_names)

Train_SIZE = 128
Test_SIZE = 128

# Redefine pickle function
def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj

# Load image
def loadimage(dataset, IMAGE_SIZE=Test_SIZE):
    images = []
    labels = []
    count =0
    for i in range(nb_classes):
        label = i
        fold = f'{folder_path}/{dataset}/{class_names[i]}'
        for file in tqdm(os.listdir(fold)):
            img_path = os.path.join(fold, file)
            image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
            # image = cv2.cvtColor(image, cv2.COLOR_2RGB)
            image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
            images.append(image)
            labels.append(label)

    images = np.array(images, dtype = 'int32')
    labels = np.array(labels, dtype = 'int32')
    return images, labels

# Load augmented image
def load_augmented_image(datasets, IMAGE_SIZE=Test_SIZE):
    images = []
    labels = []
    count =0
    ord = 0
    for dataset in datasets:
        for i in range(nb_classes):
            label = i
            fold = f'{dataset}/{folder_names[i]}'
            for file in tqdm(os.listdir(fold)):
                if dataset==datasets[0]:
                    img_path = os.path.join(fold, file)
                    image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
                    # image = cv2.cvtColor(image, cv2.COLOR_2RGB)
                    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
                    images.append(image)
                    labels.append(label)
                elif ((dataset==datasets[1]) & (len(file)>8)):
                    img_path = os.path.join(fold, file)
                    image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
                    # image = cv2.cvtColor(image, cv2.COLOR_2RGB)
                    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
                    images.append(image)
                    labels.append(label)
    images = np.array(images, dtype = 'int32')
    labels = np.array(labels, dtype = 'int32')
    return images, labels

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  infer_time_tot = 0
  infer_iter = 0
  global infer_avg_time
  infer_avg_time = 0

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    infer_iter += 1
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)
    t = time.perf_counter()
    # Run inference.
    interpreter.invoke()
    delta_t = time.perf_counter()-t
    infer_time_tot += delta_t
    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Return average inference time
  infer_avg_time = infer_time_tot/infer_iter

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy

if __name__ == 'main':
  # Parser
  parser = argparse.ArgumentParser()
  feat = parser.add_mutually_exclusive_group()
  feat.add_argument('--use_tflite', action='store_true', help='Using tflite model for evaluation')
  feat.add_argument('--use_savedmodel', action='store_true', help='Using Tensorflow SavedModel for evaluation')
  parser.add_argument('model', type=str, help='Model\'s path')
  args = parser.parse_args()
  
  # Path to models
  tflite_model_file = args.model

  '''
  Load data for train/test
  '''
  # NICS_images, NICS_labels = loadimage('train')
  test_images, test_labels = loadimage('test')

  print(len(test_images))

  # Re-load data from pkl
  # train_images = _load_pkl('/content/drive/MyDrive/MICA/BACaff_images')
  # train_labels = _load_pkl('/content/drive/MyDrive/MICA/BACaff_labels')

  # Check number of data
  # print("Training images: {}".format(train_images.shape))
  # print("Training labels: {}".format(train_labels.shape))
  # print("Validation images: {}".format(val_images.shape))
  # print("Validation labels: {}".format(val_labels.shape))
  print("Test images: {}".format(test_images.shape))
  print("Test labels: {}".format(test_labels.shape))

  '''
  Load model
  '''
  interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
  interpreter.allocate_tensors()

  eval_res = evaluate_model(interpreter)

  print("Accuracy: ", eval_res*100, "%")
  print(f'Elapsed time: %f', infer_avg_time)
  # print("Accuracy (quantized model): ", evaluate_model(interpreter_quant)*100, "%")