# test_on_AWS_EC2

Assume GPU limit is permitted. The following steps can run the model on tensorflow-gpu in the prepared image in EC2 instance.  

* Launch instance
  - Actions -> Instance states -> start / stop

* Initialize the setting
  - Login with `Git bash`
    - USER: ssh carnd@`<public ip>`
    - PSW: carnd
  - Download the test files `git clone https://github.com/YouYueHuang/test_on_AWS_EC2.git`
  - Update the test files `git pull https://github.com/YouYueHuang/test_on_AWS_EC2.git`

* Launch jupyter notebook 
  - source activate carnd-term1
  - jupyter notebook `<name of the notebook>`.ipynb

* Access jupyter notebook on local machine browser
  - http://`<public ip>`:8888/?token=3156e...
  - `<public ip>`:8888/tree

* Install python packages 
  - opencv
    - `pip install opencv-python`
  - tensorflow-gpu
    - `pip uninstall tensorflow-gpu`
    - `pip install tensorflow-gpu==0.12.1`
    - https://www.youtube.com/watch?v=cL05xtTocmY
  - keras
    - `conda install -c conda-forge keras`
  - Other
    - `pip install tqdm`

* Install CUDA and cuDNN
  - check architecture: uname -m
  - https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64
  - https://developer.nvidia.com/rdp/cudnn-download

* For incompatible of keras and tensorflow-gpu
  - check the text editor `sudo update-alternatives --config editor`
  - open file with nano `nano /path/to/filename`
  - check the path of keras package with  `os.path.abspath(keras.__file__)`
  - change the k value in `losses.py` and `metrics.py` from -1 to a possitive number (e.g, 1)

* There are 3 ways to transfer files between local machine and Amazon instance
  - [Dowload file with Filezila or scp](http://angus.readthedocs.io/en/2014/amazon/transfer-files-between-instance.html)
    - This approach need to generate key-pair pem file when setting the instance.
  - [Upload project on Github](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/)
    - This approach need to remove the files of size over 100 MB.
    - Generate the token in Github
  - Directly download file on `Jupyter`