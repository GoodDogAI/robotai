from msilib.schema import Directory


This is the main brains behind the robotai system.

# Setup
Make sure that you have a python environment setup,  I like to install miniconda, and
setup a "robotai" environment for it.
`conda create -n robotai python=3.9`

Then, install the basic system dependencies, this will work on Ubuntu generally
This command will also install the python environment dependencies via a requirements.txt file
`./setup_env.sh`


# Building
Make a new build Directory
`mkdir build`
`cd build`

Configure cmake
`cmake ..`

Build the code
`cmake --build .`


