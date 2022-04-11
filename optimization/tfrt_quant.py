#!/mnt/f/MICA/.endo_env/bin/python3

import argparse
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


parser = argparse.ArgumentParser()
parser.add_argument('--tf2',    type='str', help='TF2.0 Saved Model dir')
parser.add_argument('--prec',   type='str', help='Precision type (FP32/FP16/INT8)')
parser.add_argument('--output', type='str', help='Output TRT file dir')
args = parser.parse_args()

FP = args.prec
output_saved_model_dir = args.output

params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params = params._replace(precision_mode=FP)
converter = trt.TrtGraphConver(
                                input_saved_model_dir=args.tf2,
                                conversion_params=params
                                )
converter.convert()


converter.convert(calibration_input_fn=representative_dataset_gen)


converter.build(input_fn=representative_dataset_gen)

# Save the TRT engine and the engines.
converter.save(output_saved_model_dir)
