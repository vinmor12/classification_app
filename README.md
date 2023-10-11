# Classification Application

OpenVINO 2019 R1 application to classify images, videos or data streams from webcams

It can be used with any IR classification model.
It has been tested on Ubuntu 16.04 LTS with the following models:
  - squeezenet1.1
  - alexnet
  - googlenet-v1
  - googlenet-v2

You can find these models both in FP32 format (for CPU inference) and in FP16 format (for FPGA inference) in "models/classification" and their labels files in "data/labels".

Usage
- 
python3 classification_script.py [-h] -m MODEL [-im INPUT_MODE] [-i INPUT] [-ni NUMBER_ITER] [-pp PLUGIN_DIR] [-l CPU_EXTENSION] [-w WAIT] [-d DEVICE] [--labels LABELS] [-nt NUMBER_TOP]  
  
The arguments in square brackets are optional.  
  
Options:  
-
  -h, --help  
  Show help message and exit.  
    
  -m MODEL, --model MODEL  
  Required. Path to an .xml file with a trained model.  
    
  -im INPUT_MODE, --input_mode INPUT_MODE  
  Optional. Specify the input mode: IMG, VID or CAM. Default value is CAM.  
  
  -i INPUT, --input INPUT  
  Required for IMG and VID mode. Path to an image or video files.  
                          
  -ni NUMBER_ITER, --number_iter NUMBER_ITER  
  Optional for IMG mode. Number of iteration to compute inference time. Default value is 1.  
                          
  -pp PLUGIN_DIR, --plugin_dir PLUGIN_DIR  
  Optional. Path to a plugin folder.  
                          
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION  
   Optional. Required for CPU custom layers. MKLDNN (CPU)-targeted custom layers.Absolute path to
   a shared library with the kernels implementations.  
                          
  -w WAIT, --wait WAIT    
  Optional for CAM or VID mode. Milliseconds of wait per frame. Default value is 25.  
    
  -d DEVICE, --device DEVICE  
   Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL, MYRIAD or
   HETERO.Default value is CPU.  
                          
  --labels LABELS         
  Optional. Path to a labels mapping file.  
    
  -nt NUMBER_TOP, --number_top NUMBER_TOP  
  Optional. Number of top results (min 1 - max 10). Default value is 5.  
                          
                        
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
  - inizialize OpenVINO environment (check the correct path of your "setupvars.sh" file)
  ```
  source /opt/intel/2019_r1/openvino/bin/setupvars.sh -pyver 3.5
  ```
  - see the available options
  ```
  python3 classification_script.py -h
  ```
  - run the application to classify an image
  ```
  python3 classification_script.py \
  -im IMG \
  -m models/classification/googlenet/v1/IR/googlenet-v1.xml \
  -i data/images/aereo.png \
  --labels data/labels/imagenet_2012.txt \
  -nt 4 \
  -ni 100
  ```
  - run the application to classify frames of a video
  ```
  python3 classification_script.py \
  -im VID \
  -m models/classification/googlenet/v1/IR/googlenet-v1.xml \
  -i data/videos/desk_video.mp4 \
  --labels data/labels/imagenet_2012.txt \
  -nt 4 \
  -w 200
  ```
  - run the application to classify data streams from webcam
  ```
  python3 classification_script.py \
  -im CAM \
  -m models/classification/googlenet/v1/IR/googlenet-v1.xml \
  --labels data/labels/imagenet_2012.txt \
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
  - inizialize the environment
  ```
  source setup_board_tsp.sh
  ```
  - check the environment (the “DIAGNOSTIC_PASSED” represents the environment setup is successful)
  ```
  aocl diagnose
  ```
  - get the repository and switch to "classification_app" folder
  - see the available options
  ```
  python3 classification_script.py -h
  ```
  - run the application to classify an image on FPGA
  ```
  python3 classification_script.py \
  -im IMG \
  -m models/classification/squeezenet/1.1/IR_FP16/squeezenet1.1.xml \
  -i data/images/aereo.png \
  --labels data/labels/imagenet_2012.txt \
  -d HETERO:FPGA,CPU \
  -nt 4 \
  -ni 100
  ```
  - run the application to classify frames of a video on FPGA
  ```
  python3 classification_script.py \
  -im VID \
  -m models/classification/squeezenet/1.1/IR_FP16/squeezenet1.1.xml \
  -i data/videos/desk_video.mp4 \
  --labels data/labels/imagenet_2012.txt \
  -d HETERO:FPGA,CPU \
  -nt 4 \
  -w 200
  ```
  - run the application to classify data streams from webcam on FPGA
  ```
  python3 classification_script.py \
  -im CAM \
  -m models/classification/squeezenet/1.1/IR_FP16/squeezenet1.1.xml \
  --labels data/labels/imagenet_2012.txt \
  -d HETERO:FPGA,CPU \
  -nt 4 \
  -w 200
  ```

You can also pause the classification of video frames with the "p" key and can resume it with the "c" key.
You can exit with the "q" key.

![classification](https://raw.githubusercontent.com/vinmor12/classification_app/main/data/results/test1.png)
