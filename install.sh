pip3 install scikit-image imageio scipy opencv-python pillow==6.2.1
pip3 install 'git+https://github.com/haoxusci/pytorch_zoo@master#egg=pytorch_zoo'
python3 setup.py build_ext --inplace
pip3 install . --upgrade
