#!/usr/bin/env python

"""
CLASSIFICATION
Vincenzo.

Script to classify images, videos or data streams from webcams and 
to evaluate inference time from different classification models.

Example of use:
>> sudo su
>> source /opt/intel/2019_r1/openvino/bin/setupvars.sh -pyver 3.5
>> cd <path/to/application/folder>
>> python3 classification_script.py \
-im IMG \
-m /models/classification/googlenet/v1/IR/googlenet-v1.xml \
-i /data/images/aereo.png
--labels /data/labels/imagenet_2012.txt \
-ni 100

Workflow:
1 - set target device
2 - load HW plug-in 
3 - read IR model 
4 - allocate input & output blobs 
5 - load model to plug-in 
6 - read data into input blob 
7 - inference 
8 - process data from output blob
"""


# INCLUDE LIBRARIES

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
import time
from datetime import datetime
from openvino.inference_engine import IENetwork, IEPlugin


# INPUT ARGUMENT

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", 
                      required=True, type=str)
    args.add_argument("-im", "--input_mode", help="Optional. Specify the input mode: IMG, VID or CAM. Default value is CAM.",                        
                      default="CAM", type=str)
    args.add_argument("-i", "--input", help="Required for IMG and VID mode. Path to an image or video files.", 
                      default=None, type=str)
    args.add_argument("-ni", "--number_iter", help="Optional for IMG mode. Number of iteration to compute inference time"
                      "Default value is 1.",
                      default=None, type=str)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder.", 
                      default=None, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. MKLDNN (CPU)-targeted custom layers."
                      "Absolute path to a shared library with the kernels implementations.", 
                      type=str, default=None)
    args.add_argument("-w", "--wait", help="Optional for CAM or VID mode. Milliseconds of wait per frame. Default value is 25.",   
                      type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO."
                      "Default value is CPU.",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file.", 
                      default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results (min 1 - max 10). Default value is 5.", 
                      default=5, type=int)

    print("\nWELCOME TO CLASSIFICATION\n")
    time.sleep(2)
    return parser


# INFERENCE AND PROCESSING OUTPUT FUNCTION

def inf_proc(nt_in,image,net,exec_net,input_blob,out_blob,labels_map):
    
    # Process Input Image
    n, c, h, w = net.inputs[input_blob].shape
    image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))

    # Inference
    start_time = datetime.now().timestamp()*1000
    res = exec_net.infer(inputs={input_blob: image})
    end_time = datetime.now().timestamp()*1000

    # Process Output from Output Blob
    res = res[out_blob]
    max_index = np.argmax(res)
    max_prob = max(max(res))
    inference_time = end_time-start_time
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-nt_in:][::-1]
        index = 0
        out = np.full(shape = nt_in, fill_value = None)
        for id in top_ind:
            det_label = labels_map[id] if labels_map else "{}".format(id)
            label_length = len(det_label)
            if label_length>18:
                det_label = det_label[0:18]
                label_length = len(det_label)
            out[index] = str(det_label)+str(' ' * (20-label_length))+str(round(probs[id]*100,2))
            index = index+1
  
    return out, inference_time


# MAIN FUNCTION
    
def main():  
  
    # Read input arguments 
    
    # Clean and Resize window console
    # sys.stdout.write("\x1b[8;35;60t") # resize window console
    os.system("clear")
    time.sleep(1)
    # Load Input Data
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # Pre-process input arguments

    # Settings and Info about CAM mode
    if args.input_mode!="IMG" and args.input_mode!="VID":
        args.input_mode="CAM"
        if args.input is not None:
            log.info("Input Image/Video is not required for CAM mode")
            time.sleep(1)
        if args.number_iter is not None:
            args.number_iter = int(1)
            log.info("Number of image is not required for CAM mode")
            time.sleep(1)
        else:
            args.number_iter = int(1)
        if args.wait is not None:
            args.wait = int(args.wait)
        else:
            args.wait = int(25)
    elif args.input_mode=="IMG":
    # Settings and Info about IMG mode
        if args.number_iter is None:
            args.number_iter = int(1)
            log.info("Min number of image: {}".format(args.number_iter))
            time.sleep(1)
        else:
            args.number_iter = int(args.number_iter)
        if args.wait is not None:
            args.wait = int(1)
            log.info("Wait per frame is not required for IMG mode!")
            time.sleep(1)
        else:
            args.wait = int(1)
    # Setting and Info about VID mode
    elif args.input_mode=="VID":
        args.input_mode="VID"
        if args.number_iter is not None:
            args.number_iter = int(1)
            log.info("Number of image is not required for CAM mode")
            time.sleep(1)
        else:
            args.number_iter = int(1)
        if args.wait is not None:
            args.wait = int(args.wait)
        else:
            args.wait = int(25)

    # Settings and General Info
    log.info("Input Mode: {}".format(args.input_mode))
    time.sleep(1)
    if args.labels is None:
        log.info("Path to a labels mapping file is not specified!")
        time.sleep(1)
    if args.number_top>=10:
        args.number_top = 10
        log.info("Number of top results has been clipped to 10 for correct display")
        time.sleep(1)
    else:
        log.info("Number of top results: {}".format(args.number_top))
        time.sleep(1)
    
    # Set Target Device and Load HW Plugin

    device_target = args.device
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    
    # Read an IR Model

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    model_name = os.path.splitext(os.path.basename(model_bin))[0]
    net = IENetwork(model=model_xml, weights=model_bin)
    
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    
    assert len(net.inputs.keys()) == 1, "Application supports only single input topologies"
    assert len(net.outputs) == 1, "Application supports only single output topologies"
    
    # Allocate Input and Output Blobs

    log.info("Preparing input/outpu blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    
    # Load Model to Plugin

    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    
    # Process Data
    
    # Load labels map
    if args.labels is not None:
        log.info("Loading labels map")
        with open(args.labels, 'r') as f:
                labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    
    # Iniziale variables
    log.info("Inizialize variables")
    enable = 1
    pause = int(args.wait)
    out = ["null"]*args.number_top
    inference_time = 0
    flag = 1
    max_prob = 0
    new_frame_time = 0
    prev_frame_time = 0
    old_time = 0
    sum_time = 0
    if args.input_mode!="IMG":
        if args.input_mode=="CAM":
            vid = cv2.VideoCapture(0)
        if args.input_mode=="VID":
            vid = cv2.VideoCapture(args.input)
    
    # Set font text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Loop
    log.info("Start loop")
    time.sleep(2)
    while (True):
        if enable >= 1:

            # clean window console
            os.system("clear")
        
            # print model name and number of frame/image
            print("IR model: ",model_name,"\nframe N. ",flag-1)
        
            # read and process input file from cam or image
            if args.input_mode=="IMG":
                frame = cv2.imread(args.input)
                str_fps = "N: "+str(round(flag-1,2))      
            else:
                ret, frame = vid.read()
                new_frame_time = datetime.now().timestamp()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                str_fps = "FPS: "+str(round(fps,2)) 
                print("FPS", round(fps,2))
    
            # prepare and print text on windows console
            classid_str = "class"
            probability_str = "probability"
            print("device: ", device_target,"\ninference time: ", round(inference_time,2),"ms\n\n"+classid_str+\
        str(' ' * (20-len(classid_str)))+probability_str)
            print(str('-' * 18)+" "+str('-' * len(probability_str)))
            for i,r in enumerate(out):
                print(out[i],"\n")
            log.info("COMPUTING AVERAGE INFERENCE TIME")
            if args.input_mode!="IMG":
                log.info("Press 'q' to Exit or 'p' to Pause!")
            else:
                print("."*30+"#", end = "\r")
                print("#"*int(flag*30/args.number_iter),end = "\r")
            
            # prepare and draw text on image
            text = ["IR model: "+str(model_name),str_fps," ","device: "+str(device_target),\
        "inference time: "+str(round(inference_time,2))+"ms"," ",str(classid_str)+str(' ' * (18-len(classid_str)))+\
        str(probability_str),str('-' * (len(probability_str)+len(classid_str)))]
            offset = 10
            for raw in text:
                offset = offset + 25
                cv2.putText(frame,raw,(10,offset),font,0.5,(10,255,10),2)
            store_off = offset
            for raw in out:
                offset = offset + 25
                cv2.putText(frame,raw[0:20],(10,offset),font,0.5,(255,255,10),2)
            offset = store_off
            for raw in out:
                offset = offset + 25
                cv2.putText(frame,raw[20:24],(180,offset),font,0.5,(0,0,255),2)
        
            # inference (with "inf_proc" function defined before)
            if args.input_mode != "IMG":
                if ret == True:    
                    out, inference_time = inf_proc(args.number_top,frame,net,exec_net,input_blob,out_blob,labels_map)
                else:
                    break
            else:
                out, inference_time = inf_proc(args.number_top,frame,net,exec_net,input_blob,out_blob,labels_map)

            
            # evaluate average time of inference  
            sum_time = sum_time + inference_time  
            inference_time = sum_time/(flag)
     
        # show loading bar for evaluating of average inference time, show resize image and exit
        if args.input_mode=="IMG":
            print("."*30+"#", end = "\r")
            print("#"*int(flag*30/args.number_iter),end = "\r")
            if flag > args.number_iter:
                cv2.imshow(args.input_mode,cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA))
                log.info("FINISH! Press Any Key to Exit!")
                cv2.waitKey(0)
                break
        else:
        # show frame from cam and allow the system pause
            if flag > 1:
                cv2.imshow(args.input_mode,cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA))
                #cv2.imshow(args.input_mode,frame)
            if cv2.waitKey(pause) & 0xFF == ord('q'):
                break
            if enable == 1:
                if cv2.waitKey(pause) & 0xFF == ord('p'):
                    enable = 0
                    print("PAUSE! press 'c' to continue ...", end="\r")               
            if enable == 0:
                if cv2.waitKey(pause) & 0xFF == ord('c'):
                    for i in range (0,3+int(pause/1000)):
                        print("SYSTEM REMUSMES in", 3+int(pause/1000)-i,"s... Release the key!", end = "\r") 
                        time.sleep(1)
                    enable = 1

        # increase frame counter
        if enable!= 0:
            flag = flag+1

    # destroy all windows
    if args.input_mode!="IMG": 
        vid.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    sys.exit(main() or 0)

