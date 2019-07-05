# Using CLion with ML-CPP

CLion 2018.2 has added support for compilation database project format. This tutorial shows how to create a compilation 
database for the `ml-cpp` project and integrate it with CLion.

## Installing Prerequisites

### CLion

You need to install current [CLion](https://www.jetbrains.com/clion/) (version 2018.2 or later) with following plugins:
* Compilation Database
* File Watchers
* Makefile support

You can install the plugins either at the first start during the initialization dialog or later through 
**Settings/Preferences->Plugins**.  

### Compilation Database

Install Python module [compiledb](https://github.com/nickdiego/compiledb) from PyPi:
```
pip install compiledb
```

It is better to install the module on the system level, since it will create an executable `compiledb` which can be 
easier integrated into CLion.

##  Create Compilation Database and Load in CLion

## Create Custom Targets for Unit Tests

## Load settings for the repository

https://www.jetbrains.com/help/clion/managing-makefile-projects.html
https://www.jetbrains.com/help/clion/compilation-database.html
https://www.jetbrains.com/help/clion/custom-build-targets.html
https://www.jetbrains.com/help/phpstorm/sharing-your-ide-settings.html
