"""
Use this file if you are building a Real-time Gesture Recognition application. This python module will accept image frames from your client, process it and provide you output.

The frames are to be sent to the server in the following manner:

* First send the size of the image in bytes.
* Next, send the complete image in a byte array.

This module also uses Text-to-speech module for speech output at the server end. 
You can turn it off by setting the `no_speech_output` false, or by passing `--nospeech 1` parameter while running the program.
"""

import socket, struct, atexit, timeit
import sys, os, distutils, argparse, pyttsx3
import cv2, imutils
import netifaces as ni, numpy as np
from server_utils import addToQueue, displayTextOnWindow, getConsistentSign
from sys import getsizeof

import silatra

parser = argparse.ArgumentParser(description='Main Entry Point')
parser.add_argument('--port', help='Opens Silatra server at specified port number.')
parser.add_argument('--mode', help='Default is Gesture mode. --mode 1 for Hand pose recognition')
parser.add_argument('--nostabilize', help='Specify --nostabilize 1 to not use Object stabilization')
parser.add_argument('--nospeech', help='Specify --nospeech 1 to not use speech output')
args = parser.parse_args()


if not args.port: port = int(input('Enter port number to start server: '))
else: port = int(args.port)

if not args.mode: mode = 'Gesture'
else: mode = 'Hand pose recognition'

if not args.nostabilize: use_stabilization = True
else: use_stabilization = False

if not args.nospeech: no_speech_output = False
else: no_speech_output = True

if not no_speech_output: engine = pyttsx3.init()

ip_address = ni.ifaddresses('wlp2s0')[ni.AF_INET][0]['addr']

print('Starting Silatra server. Use IP address as %s and port as %d at your client site.'%(ip_address, port))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', port))
s.listen(1)
print('Waiting for connections...')
client, address = s.accept()
print('Connected to client with IP Address %s'%ip_address)

if mode == 'Gesture': unknown_gesture = silatra.Gesture(use_stabilization)

no_of_frames = 0
minNoOfFramesBeforeGestureRecogStart = 70

while True:
    buf = client.recv(4)
    print('Received message of size %d bytes'%(getsizeof(buf)))
    size = struct.unpack('!i', buf)[0]
    print("receiving image of size: %s bytes" % size)
    
    data = client.recv(size, socket.MSG_WAITALL)
    numpy_array = np.fromstring(data, np.uint8)
    img_np = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    try: img_np = imutils.rotate_bound(img_np, 90)
    except AttributeError: break
    img_np = cv2.resize(img_np, (0,0), fx=0.7, fy=0.7)

    cv2.imshow('Frames received', img_np)
    no_of_frames += 1
    
    if size > 0:
        if mode == 'Hand pose recognition':
            recognised_pose = silatra.recognise_hand_pose(img_np, using_stabilization=use_stabilization, no_of_frames=no_of_frames)
            consistent_pose = str(getConsistentSign(recognised_pose))
            if consistent_pose == '-1': consistent_pose = 'No hand pose in image'
            displayTextOnWindow('Recognised Hand pose', consistent_pose)
            client.send(consistent_pose.encode('ascii'))

            if not no_speech_output:
                engine.say(consistent_pose)
                engine.runAndWait()

        if mode == 'Gesture':
            if no_of_frames == minNoOfFramesBeforeGestureRecogStart - 10:
                op1 = "Model ready to recognize\r\n"
            elif no_of_frames == minNoOfFramesBeforeGestureRecogStart:
                op1 = "Start gesture\r\n"
            elif len(observations) == 0:
                pass
            client.send(op1.encode('ascii'))

            if no_of_frames > minNoOfFramesBeforeGestureRecogStart: unknown_gesture.add_frame(img_np)

    if size == 0:
        if mode == 'Hand pose recognition':
            client.send("QUIT\r\n".encode('ascii'))
            break
        elif mode == 'Gesture':     # End of gesture
           if len(unknown_gesture.get_observations()) > 0:
               recognised_gesture = unknown_gesture.classify_gesturei()
               displayTextOnWindow('Recognised Gesture',recognised_gesture)
               client.send(recognised_gesture.encode('ascii'))

               if not no_speech_output:
                   engine.say(recognised_gesture)
                   engine.runAndWait()
           client.send("QUIT\r\n".encode('ascii'))
           break

print('Silatra server stopped!')
def cleaners():
    s.close()
    cv2.destroyAllWindows()

atexit.register(cleaners)
