# Classification Application

OpenVINO 2019 R1 application to classify images, videos or data streams from webcams

It can be used with any IR classification model.
It has been tested on Ubuntu 16.04 LTS with the following models:
  - squeezenet1.1
  - alexnet
  - googlenet-v1
  - googlenet-v2

You can find these models both in FP32 format (for CPU inference) and in FP16 format (for FPGA inference) in "models/classification".

You can use this application in the following way:
  - open a new Terminal and clone the repository
    ```
    git clone https://github.com/vinmor12/classification_app.git
    ```
