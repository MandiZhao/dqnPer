# with PER, CS294-112 HW3: Q-Learning

## How I ran my code

### cd to project folder and activate/create virtual env
```
virtualenv venv
source venv/bin/activate
pip install tensorflow_gpu==1.12
pip install -r requirements.txt
```
### After that if an error about not detecting ffmpeg shows, 
### manually install ffmeg without sudo

refer to this [link]: (https://www.johnvansickle.com/ffmpeg/faq/)
```
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
tar xvf ffmpeg-git-amd64-static.tar.xz
```
You will get a folder named ffmpeg-git-[today’s date]-amd64-static, 
then move ffmpeg and ffprobe inside to /venv/bin:

```
mv ffmpeg-git-[today’s date]-amd64-static/ffmpeg ffmpeg-git-20180203-amd64-static/ffprobe /venv/bin/

```


### Now it should run
#### To alternate between vanilla and PER:
Inside run_dqn_atari.py:
line 139 in atari_learn(): set plain=True for vanilla and plain=False for PER

#### Finally run
```
$ python run_dqn_atari.py
```


## Bugs
The code should print out the shape of processed frames from atari_wrappers.py
Mine is still not working properly and shows (210, 160, 3)
So in dqn.py (and dqn_plain.py) I manually stacked and reshaped the observasions from env.


Output from $pip list: 

pip list
Package                 Version      Location                          
------------------- -------- ----------------------------------
absl-py             0.7.1    
astor               0.8.0    
atari-py            0.2.6    
baselines           0.1.6    /home/mannndy/github-hw3/baselines
Click               7.0      
cloudpickle         1.2.1    
ffmpeg              1.4      
ffmpeg-python       0.2.0    
future              0.17.1   
gast                0.2.2    
grpcio              1.22.0   
gym                 0.14.0   
h5py                2.9.0    
joblib              0.13.2   
Keras-Applications  1.0.8    
Keras-Preprocessing 1.1.0    
Markdown            3.1.1    
numpy               1.17.0   
opencv-python       4.1.0.25 
Pillow              6.1.0    
pip                 19.2.1   
protobuf            3.9.0    
pyglet              1.3.2    
scipy               1.3.0    
setuptools          41.0.1   
six                 1.12.0   
tensorboard         1.12.2   
tensorflow-gpu      1.12.0   
termcolor           1.1.0    
tqdm                4.32.2   
updates             0.1.7.1  
Werkzeug            0.15.5   
wheel               0.33.4   


------------


[HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) 

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.
