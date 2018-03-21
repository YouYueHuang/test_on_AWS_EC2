# test_on_AWS_EC2

* Launch instance
  - Actions -> Instance states -> start / stop

* In `Git bash`, type
  - USER: ssh carnd@`<public ip>`
  - PSW: carnd

* Download
`git clone https://github.com/YouYueHuang/test_on_AWS_EC2.git`

* Update
`git pull https://github.com/YouYueHuang/test_on_AWS_EC2.git`

* source activate carnd-term1
* jupyter notebook `<name of the notebook>`.ipynb

* on local machine browser
  - http://`<public ip>`:8888/?token=3156e...
  - `<public ip>`:8888/tree
  - jupyter notebook --port 8888

* install opencv
  - `pip install opencv-python`

* install tensorflow-gpu
  - `pip uninstall tensorflow-gpu`
  - `pip install tensorflow-gpu==0.12.1`
  - https://www.youtube.com/watch?v=cL05xtTocmY

* install keras
  - `conda install -c conda-forge keras`

* other
  - `pip install tqdm`

* install CUDA and cuDNN
  - check architecture: uname -m
  - https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64
  - https://developer.nvidia.com/rdp/cudnn-download

* check the text editor
  - `sudo update-alternatives --config editor`

* open file with nano
  - `nano /path/to/filename`

* check package path
  - `print (os.path.abspath(my_module.__file__))`

* change `losses.py` and `metrics.py`

* transferring files between your laptop and Amazon instance
http://angus.readthedocs.io/en/2014/amazon/transfer-files-between-instance.html

pip install pydot

