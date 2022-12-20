# Installation
Full documentation on how to install pysc2 can be found [here](https://github.com/Blizzard/s2client-proto/blob/master/docs/linux.md).
It is recommended to run the simulation on Linux, because on Windows/Mac the
visualization can't be turned off.

## Install Starcraft 2 Client
Download the zip file and extract it to the installation location.
Remember the installation path, it needs to be set as an environment variable later.

[Full setup documenation](https://github.com/Blizzard/s2client-proto/blob/master/docs/linux.md)

## Install PySC2 pip package
This package provides a python client to interact with the Starcraft 2 API.
The package is called *pysc2* and can be installed with:

```shell
pip install pysc2
```

It is best to install the package in a virtual environment, e.g. with *pipenv* to keep all dependencies encapsulated.

## Configure environment by setting environment variables
The environment specific settings are configured through environment variables. The settings themselves shouldn't be pushed to git as they contain information about the local environment including local paths or protected data like credentials.

When developing locally on Linux you can use *direnv* to set the environment variables automatically, when you are in the project directory or one of its subdirectories.

You can find a sample of the required environment variables in [*env-sample.sh*](../env-sample.sh).
