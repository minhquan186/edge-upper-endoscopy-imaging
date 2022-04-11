import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

onnx_path = 'Models/mobilenetv2_og.onnx'
saved_path = 'Models/mobilenetv2_onnxruntime_quant.onnx'
# Load the onnx model
model = onnx.load(onnx_path)
# Quantize and save
quantize_dynamic(   onnx_path, 
                    saved_path,
                    weight_type=QuantType.QInt8
                )
# Prompt successfully quantized
print(f"Quantized model successfully saved to:{saved_path}")
