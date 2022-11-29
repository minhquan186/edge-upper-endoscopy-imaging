#!/mnt/f/MICA/.endo_env/bin/python3

# Import libs
import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
import tensorflow as tf
import time
import argparse
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Setup
# folder_path="/home/mica/edge-upper-endoscopy-imaging/data_NICS"
# folder_path="/mnt/f/MICA/data_NICS"
folder_path="/home/quan/MICA/data_NICS"
class_names = [ '1 Hầu họng', '2 Thực quản', '3 Tâm vị', '4 Thân vị', '5 Phình vị', '6 Hang vị','7 Bờ cong lớn','8 Bờ cong nhỏ','9 Hành tá tràng','10 Tá tràng']
folder_names = ['Vung hau hong', 'Thuc quan', 'Tam vi', 'Than vi','Phinh vi', 'Hang vi','Bo cong lon','Bo cong nho','Hanh ta trang','Ta trang']
nb_classes = len(class_names)

Train_SIZE = 128
Test_SIZE = 128

# One hot encoding
def labelEncoder(label):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label)


    onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoder = OneHotEncoder(sparse=False,dtype=np.uint8)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    encoded_label = onehot_encoded

    return encoded_label

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

def load_test_data():
  '''
  Load data for train/test
  '''
  # NICS_images, NICS_labels = loadimage('train')
  test_images, test_labels = loadimage('test')
  # test_labels = labelEncoder(test_labels)
  print(len(test_images))

  # Check number of data
  print("Test images: {}".format(test_images.shape))
  print("Test labels: {}".format(test_labels.shape))


# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter, test_images):
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
    # Post-processing: remove batch dimension and find the digit with highest probability.
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

if __name__ == '__main__':
  # Parser
  parser = argparse.ArgumentParser()

  feat = parser.add_mutually_exclusive_group()
  feat.add_argument('--use_tflite', dest='tflite', action='store_true', help='Using tflite model for evaluation')
  feat.add_argument('--use_savedmodel', dest='tf2', action='store_true', help='Using Tensorflow SavedModel for evaluation')
  feat.add_argument('--use_tensorrt', dest='trt', action='store_true', help='Using TensorRT SavedModel for evaluation')
  
  parser.add_argument('model', type=str, help='Model\'s path')

  args = parser.parse_args()  
  model_path = args.model

  '''
  Load model
  '''
  if args.tflite:
    load_test_data()
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    eval_res = evaluate_model(interpreter, test_images)
    print("Accuracy: ", eval_res*100, "%")
    print(f'Elapsed time: %f', infer_avg_time)
    # print("Accuracy (quantized model): ", evaluate_model(interpreter_quant)*100, "%")
  elif args.tf2:
    load_test_data()
    # Load Tensorflow SavedModel format
    loaded = tf.keras.models.load_model(model_path)
    # print(loaded.summary())
    t1 = time.perf_counter()
    # res = loaded.predict(test_images)
    # res = np.argmax(res,axis=1)
    res = loaded.evaluate(test_images, test_labels)
    elapsed = time.perf_counter() - t1
    print('Loss: {:.4f}'.format(res[0]))
    print('Accuracy: {:.4f}%'.format(res[1]*100))
    print('Elapsed time: {:.4f}s'.format(elapsed))
    # print(res[:15])
  elif args.trt:
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    # print(signature_keys)
 
    infer = saved_model_loaded.signatures['serving_default']
    # print(infer.structured_outputs)
    for i in range(nb_classes):
        avg_time = 0
        total_time = 0
        iterations = 0
        correct_class_count = 0
 
        folder = f'{folder_path}/test/{class_names[i]}'
    
        for file in os.listdir(folder):
          iterations += 1
          img_path = os.path.join(folder, file)
          img = image.load_img(img_path, target_size=(128, 128))
          
          x = image.img_to_array(img)
          x = np.expand_dims(x, axis=0)
          x = preprocess_input(x)
          x = tf.constant(x)
          
          start = time.perf_counter()
          labeling = infer(x)
          end = time.perf_counter()
          elapsed_time = end - start
          total_time += elapsed_time
          
          predictions = next(iter(labeling.values())).numpy()
          res = np.argmax(predictions)
          if (res == i):
            correct_class_count+=1
        
        avg_time = total_time / iterations
        print(f'Average inference time: {avg_time}')
        acc = (correct_class_count / iterations) * 100
        print(f'Accuracy: {acc}%')