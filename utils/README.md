# remgb u2net

https://github.com/danielgatis/rembg?tab=readme-ov-file#models

目前google drive下载有问题，手动下载https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab ，把u2net.onnx放到/root/.u2net/u2net.onnx

Downloading data from 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx' to file '/root/.u2net/u2net.onnx'.

for human: https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx

``` python
if not os.path.exists("/root/.u2net/u2net.onnx"):
    os.makedirs("~/.u2net")
    os.system("wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -p ~/.u2net/")
```
