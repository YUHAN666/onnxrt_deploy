# onnxrt_deploy
使用onnxRt CPU 部署tensorflow .pb模型

使用3个模型做OCR，第一个模型从背景中分割出字符块，第二个模型将字符块分割成一个个字符，第三个模型对每个字符做判断
