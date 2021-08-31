# onnxrt_deploy
使用onnxRt CPU 部署tensorflow模型

需要使用tensorflow simple_save将模型保存为带参数的pb模型，再转化为onnx模型进行部署

python -m tf2onnx.convert --saved-model E:/CODES/tensorflow_ocr/chip_pbmodel_1/ --output E:/CODES/tensorflow_ocr/chip_pbmodel_1/model.onnx --inputs image_input:0 --outputs dbnet/proba3_sigmoid:0

使用3个模型做OCR，第一个模型从背景中分割出字符块，第二个模型将字符块分割成一个个字符，第三个模型对每个字符做判断
