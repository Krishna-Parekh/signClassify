
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['keras==2.2.4',
                     'Keras-Preprocessing==1.0.5',
                     #'keras-utils==1.0.13',
                     'Pillow==9.0.1',
                     'h5py==2.9.0',
                     'numpy==1.15.4',
                     'scikit-learn==0.20.2']

setup(
  name='trainer',
  version='0.1',
  author = 'Krishna Parekh',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  include_package_data=True,
  requires=[],
  description='CMLE Signature Classification',
)
