#!/bin/bash -ex
script_link="$( readlink "$BASH_SOURCE" )" || script_link="$BASH_SOURCE"
apparent_sdk_dir="${script_link%/*}"
if [ "$apparent_sdk_dir" == "$script_link" ]; then
  apparent_sdk_dir=.
fi
sdk_dir="$( command cd -P "$apparent_sdk_dir" > /dev/null && pwd -P )"

mkdir -p $sdk_dir/.dev_env

if [ ! -e $sdk_dir/.dev_env/miniconda.sh ]; then
    curl -o $sdk_dir/.dev_env/miniconda.sh \
	 -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x $sdk_dir/.dev_env/miniconda.sh
fi
if [ ! -e $sdk_dir/.dev_env/bin/conda ]; then
    $sdk_dir/.dev_env/miniconda.sh -b -u -p $sdk_dir/.dev_env
fi

$sdk_dir/.dev_env/bin/conda install -y \
  ipython \
  numpy \
  pytorch \
  torchvision \
  -c pytorch
  
$sdk_dir/.dev_env/bin/conda  clean -ya					    
source $sdk_dir/.dev_env/bin/activate $sdk_dir/.dev_env
pip install -U \
    apriltag==0.0.13 \
    opencv-python
