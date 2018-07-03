echo "Installing Silatra dependencies..."
sudo apt-get install python3-pip
sudo pip3 install numpy
sudo pip3 install sklearn
sudo pip3 install opencv-python
sudo pip3 install cmake
sudo pip3 install imutils
sudo pip3 install hmmlearn
sudo pip3 install opencv-contrib-python
sudo pip3 install dlib

# These dependencies are exclusively required for server.py
sudo apt-get install espeak     # This is required before you install pyttsx3
sudo pip3 install pyttsx3       # Python Text to Speech
sudo pip3 install netfaces      # To get IP address
