
# convert torch to TensorRT


## Pipeline

### save torch model

```python
# save weights
torch.save(model.state_dict(), path)

# load weights
state_dict = torch.load(path)
model.load_state_dict(state_dict)
```

### convert torch to onnx
```python
torch_model.eval()
input_names = ['input']
output_names = ['output']
input = torch.randn(batch_size, 3, args.image_size, args.image_size, requires_grad=True, device=device)
torch.onnx.export(torch_model,  # model being run
                  input,  # model input (or a tuple for multiple inputs)
                  onnx_model_path,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=input_names,  # the model's input names
                  output_names=output_names,  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # dynamic batch_size
                                'output':{0: 'batch_size'}},
                  # dynamic_axes = {'input': [2, 3],  # dynamic input shape
                  #                 'output':[2, 3]},
                  verbose=True) # show verbose
```
[export_model_to_onnx](./export_model_to_onnx.py)


### install tensorRT
[reference blog](https://zhuanlan.zhihu.com/p/64053177)
* download tensorRT
    [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)

check you system version
```shell script
$ cat /etc/issue
Ubuntu 14.04.5 LTS \n \l
 
$ cat /usr/local/cuda/version.txt
CUDA Version 8.0.61
```


### generate engine with onnx

* generate trtexec
```shell script
$ cd ~/samples/trtexec
$ make
```

* generate engine file
```shell script
/*f16*/
$ ./bin/trtexec --onnx={name}.onnx --fp16 --saveEngine={name}.engine
/*f32*/
$ ./bin/trtexec --onnx={name}.onnx --saveEngine={name}.engine

```
### test inference time 
```shell script
$ ./bin/trtexec --loadEngine={name}.engine --exportOutput={name}.trt
```

### inference with engine


## Reference 
* <https://zhuanlan.zhihu.com/p/64053177>
* <https://zhuanlan.zhihu.com/p/88318324>
