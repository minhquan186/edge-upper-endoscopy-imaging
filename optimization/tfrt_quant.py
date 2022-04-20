#!/mnt/f/MICA/.endo_env/bin/python3

import argparse
import tensorflow as tf

print(f'Imported necessary library!')
parser = argparse.ArgumentParser()
parser.add_argument('tf2',    type=str, help='TF2.0 Saved Model dir')
parser.add_argument('prec',   type=str, help='Precision type (FP32/FP16/INT8)')
parser.add_argument('output', type=str, help='Output TRT file dir')
args = parser.parse_args()

input_model_dir = args.tf2
print(f'Saved Model loaded!')
FP = args.prec
output_saved_model_dir = args.output

params = tf.experimental.tensorrt.ConversionParams(
    precision_mode=FP)
converter = tf.experimental.tensorrt.Converter(
                                input_saved_model_dir=input_model_dir,
                                conversion_params=params
                                )
converter.convert(is_dynamic_op=True)

print(f"Converting to TensorRT...")
converter.convert()

# converter.build(input_fn=representative_dataset_gen)

# Save the TRT engine and the engines.
converter.save(output_saved_model_dir)
print(f'TensorRT Model successfully converted and saved at {output_saved_model_dir}')