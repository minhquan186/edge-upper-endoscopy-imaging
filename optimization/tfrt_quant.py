#!/mnt/f/MICA/.endo_env/bin/python3

import argparse
# import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

parser = argparse.ArgumentParser()
parser.add_argument('tf2',    type=str, help='TF2.0 Saved Model dir')
parser.add_argument('prec',   type=str, help='Precision type (FP32/FP16/INT8)', default='FP32')
parser.add_argument('output', type=str, help='Output TRT file dir')
args = parser.parse_args()

input_model_dir = args.tf2
print(f'Saved Model loaded!')
precision = args.prec
output_saved_model_dir = args.output

if not os.path.exists(output_saved_model_dir):
    os.makedirs(output_saved_model_dir)

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision, 
                                                                    max_workspace_size_bytes=8000000000,
                                                                    use_calibration= precision == "INT8")

converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_model_dir,
                                    conversion_params=conversion_params)

if precision == "INT8":
    converter.convert(calibration_input_fn=calibration_data)
else:
    converter.convert()
# converter.convert()

# Save the TRT engine and the engines.
converter.save(output_saved_model_dir)

print(f'TensorRT Model successfully converted and saved at {output_saved_model_dir}')