import onnx
from onnx import numpy_helper
import numpy as np
import torch

#!##### --- config --- ######
model_path = "./Python_tools/train/outputs/model.onnx"
#!##########################




def onnx2c():
    # onnx読み込み
    model = onnx.load(model_path)
    
    #print test
    for node in model.graph.node:
        print(f"Name:{node.name}, OpType:{node.op_type}")
        print(f"Input:{node.input}, Output:{node.output}")
    print("\n")
    
    for tensor in model.graph.initializer:
        name = tensor.name
        weight_array = numpy_helper.to_array(tensor)
        print(f"Name: {name}")
        print(f"Shape: {weight_array.shape}")
        print(f"Values (head): {weight_array.flatten()[:5]}...") # 最初の5個だけ表示
        print("-" * 20)
    
    return

if __name__ == "__main__":
    onnx2c()