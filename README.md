# Classification Application

OpenVINO 2019 R1 application to classify images, videos or data streams from webcams

It can be used with any IR classification model.
It has been tested on Ubuntu 16.04 LTS with the following models:
  - squeezenet1.1
  - alexnet
  - googlenet-v1
  - googlenet-v2

You can find these models both in FP32 format (for CPU inference) and in FP16 format (for FPGA inference) in "models/classification".

Inference on CPU
  -  
You can use this application on CPU in the following way:
  - open a new Terminal and clone the repository
  ```
  git clone https://github.com/vinmor12/classification_app.git
  ```
  - switch to "classification_app" folder
  ```
  cd classification_app
  ```
  - switch to super-user
  ```
  sudo su
  ```
  - inizialize OpenVINO envarionment (check the correct path of your "setupvars.sh" file)
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

You can also pause the classification of video frames with the "p" key and can resume it with the "c" key.
You can exit with the "q" key.

Inference on FPGA with "Developer kit for OpenVINO toolkit"
  -  
If you want to use this application on FPGA with "Developer kit for OpenVINO toolkit", you can follow these commands:
  - switch to super-user
  ```
  sudo su
  ```
  - switch to "terasic_demo" path
  ```
  cd /opt/intel/2019_r1/openvino/deployment_tools/terasic_demo
  ```
  - inizialize the envarionment
  ```
  source setup_board_tsp.sh
  ```
  - check the environment (the “DIAGNOSTIC_PASSED” represents the environment setup is successful)
  ```
  aocl diagnose
  ```
  - get the repository and switch to "classification_app" folder
  - run the application to classify an image on FPGA
  ```
  python3 classification_script.py \
  -im IMG \
  -m /models/classification/squeezenet/1.1/IR_FP16/squeezenet1.1.xml \
  -i /data/images/aereo.png
  --labels /data/labels/imagenet_2012.txt \
  -d HETERO:FPGA,CPU
  -nt 4 \
  -ni 100
  ```
  - run the application to classify frames of a video on FPGA
  ```
  python3 classification_script.py \
  -im VID \
  -m /models/classification/squeezenet/1.1/IR_FP16/squeezenet1.1.xml \
  -i /data/videos/desk_video.mp4
  --labels /data/labels/imagenet_2012.txt \
  -d HETERO:FPGA,CPU
  -nt 4 \
  -w 200
  ```
  - run the application to classify data streams from webcam on FPGA
  ```
  python3 classification_script.py \
  -im CAM \
  -m /models/classification/squeezenet/1.1/IR_FP16/squeezenet1.1.xml \
  --labels /data/labels/imagenet_2012.txt \
  -d HETERO:FPGA,CPU
  -nt 4 \
  -w 200
  ```

You can also pause the classification of video frames with the "p" key and can resume it with the "c" key.
You can exit with the "q" key.

