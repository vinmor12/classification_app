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
  - switch in to "classification_app" folder
    ```
    cd classification_app
    ```
  - switch to super-user
    ```
    sudo su
    ```
  - inizialize OpenVINO envarionment
    ```
    source /opt/intel/2019_r1/openvino/bin/setupvars.sh -pyver 3.5
    ```
  - run the application to classify an image
    ```
    python3 classification_script.py \
    -im IMG \
    -m /models/classification/googlenet/v1/IR/googlenet-v1.xml \
    -i /data/images/aereo.png
    --labels /data/labels/imagenet_2012.txt \
    -nt 4 \
    -ni 100
    ```
  - run the application to classify frames of a video
    ```
    python3 classification_script.py \
    -im VID \
    -m /models/classification/googlenet/v1/IR/googlenet-v1.xml \
    -i /data/videos/desk_video.mp4
    --labels /data/labels/imagenet_2012.txt \
    -nt 4 \
    -w 200
    ```
  - run the application to classify data streams from webcam
    ```
    python3 classification_script.py \
    -im CAM \
    -m /models/classification/googlenet/v1/IR/googlenet-v1.xml \
    --labels /data/labels/imagenet_2012.txt \
    -nt 4 \
    -w 200
    ```
