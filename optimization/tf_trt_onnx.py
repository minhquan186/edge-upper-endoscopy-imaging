#!/mnt/f/MICA/.endo_env/bin/python3

import argparse
import onnx
import subprocess
import tf2onnx

parser = argparse.ArgumentParser()
parser.add_argument('saved_model', type=str, help='Tensorflow 2.0 Saved model path')
parser.add_argument('onnx', type=str, help='Path to save onnx model')
parser.add_argument('save_trt', type=str, help='TensorRT engine output path')

args = parser.parse_args()

input_model_dir = args.saved_model
onnx_path = args.onnx

subprocess.run(["python","-m","tf2onnx.convert","--opset","11","--saved-model",input_model_dir, "--output", "tmp.onnx"], check=True)

onnx_model = onnx.load('tmp.onnx')
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
onnx.save(onnx_model, onnx_path)
onnx.checker.check_model(onnx_model)

subprocess.run(["rm", "-fv", "tmp.onnx"])