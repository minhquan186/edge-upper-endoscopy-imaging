#!/mnt/f/MICA/.endo_env/bin/python3

import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from tqdm import tqdm
from onnx import numpy_helper

Test_SIZE = 128
folder_path="/mnt/f/MICA/edge-upper-endoscopy-imaging/data_NICS"
class_names = ['1 Hầu họng', '2 Thực quản', '3 Tâm vị', '4 Thân vị','5 Phình vị',
                '6 Hang vị','7 Bờ cong lớn','8 Bờ cong nhỏ','9 Hành tá tràng','10 Tá tràng']
folder_names = ['Vung hau hong', 'Thuc quan', 'Tam vi', 'Than vi','Phinh vi',
                'Hang vi','Bo cong lon','Bo cong nho','Hanh ta trang','Ta trang']
nb_classes = len(class_names)

model_dir ="/mnt/f/MICA/edge-upper-endoscopy-imaging/Models"
model=model_dir+"/mobilenetv2_og.onnx"

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

test_images, test_labels = loadimage('test')

for img in test_images:
#Preprocess the image
    data = json.dumps({'data': img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: data})
    prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
    print(prediction) 