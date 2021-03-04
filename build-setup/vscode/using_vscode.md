# Using Visual Studio Code with `ml-cpp`

Previously, we explained how to [set up CLion for `ml-cpp`](../clion/using_clion.md). You may find, that CLion has some
performance problems when indexing `ml-cpp` project: it may take very long to open the project or CLion may freeze and
fail. In this case, [Visual Studio Code](https://code.visualstudio.com/) can be an alternative. When combined with
`clangd`, it is significantly faster, while still being able to provide features like autocompletion, code navigation,
static code analysis, etc.

## Prerequisites

In this tutorial, we assume that you have `gcc` or `clang` installed as described in the
[CLion tutorial](../clion/using_clion.md). We also assume that you already created `compile_commands.json` file as
described there since `clangd` also uses it.

[Install `clangd`](https://clangd.llvm.org/installation.html). Version 11.0 had a significant performance improvement,
so it is worth going the extra mile and install the latest stable version. Install
[Visual Studio Code](https://code.visualstudio.com/) with
[clangd plugin](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd).

In **Settings**, make sure *C_Cpp:Autocomplete* and *C_Cpp:Intelli Sense Engine* from the native C++ plugin of VS Code
are disabled. Note that we don't want to disable the complete C/C++ Extension, since we are still going to use the
debugging capabilities.

Now for the `clangd` plugin, make sure that you have the correct path specified in **Settings** and *Clangd: Path*.
*Clangd: Arguments* can look something like this:

```text
-log=verbose
-pretty
--header-insertion=iwyu
--suggest-missing-includes
-j=4
--all-scopes-completion
--background-index=0
--clang-tidy
--compile-commands-dir=./
```

## User tasks

You can use [user tasks](https://code.visualstudio.com/docs/editor/tasks) to integrate VS Code with external tools. For
example, re-format the project using a the docker container with correct clang-format version or build certain parts of
the project in Debug mode. Later, we will see how we can use these tasks as dependencies for debugging.

Here is an example of user tasks specified in `tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Format code",
            "type": "shell",
            "command": "cd ${workspaceFolder} && dev-tools/docker/run_docker_clang_format.sh",
            "problemMatcher": [],
            "options": {
                "env": {
                    "PATH": "${env:PATH}",
                    "CPP_SRC_HOME": "${workspaceFolder}"
                }
            }             
        },
        {
            "label": "Build ml-cpp lib",
            "type": "shell",
            "command": "cd ${workspaceFolder} && make ML_DEBUG=1 -j6",
            "problemMatcher": [
                "$gcc"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "options": {
                "env": {
                    "CPP_SRC_HOME": "${workspaceFolder}"
                }
            }            
        },
        {
            "label": "Build ml-cpp api tests",
            "type": "shell",
            "command": "cd ${workspaceFolder}/lib/api/unittest && make ML_DEBUG=1 -j6",
            "problemMatcher": [
                "$gcc"
            ],
            "dependsOn": [
                "Build ml-cpp lib"
            ],
            "options": {
                "env": {
                    "CPP_SRC_HOME": "${workspaceFolder}"
                }
            }              
        },
        {
            "label": "Build ml-cpp core tests",
            "type": "shell",
            "command": "cd ${workspaceFolder}/lib/core/unittest && make ML_DEBUG=1 -j6",
            "problemMatcher": [
                "$gcc"
            ],
            "dependsOn": [
                "Build ml-cpp lib"
            ],
            "options": {
                "env": {
                    "CPP_SRC_HOME": "${workspaceFolder}"
                }
            }              
        },
        {
            "label": "Build ml-cpp maths tests",
            "type": "shell",
            "command": "cd ${workspaceFolder}/lib/maths/unittest && make ML_DEBUG=1 -j6",
            "problemMatcher": [
                "$gcc"
            ],
            "dependsOn": [
                "Build ml-cpp lib"
            ],
            "options": {
                "env": {
                    "CPP_SRC_HOME": "${workspaceFolder}"
                }
            }                       
        },
        {
            "label": "Build ml-cpp bin",
            "type": "shell",
            "command": "cd ${workspaceFolder}/bin && make ML_DEBUG=1 -j6",
            "problemMatcher": [
                "$gcc"
            ],
            "dependsOn": [
                "Build ml-cpp lib"
            ],
            "options": {
                "env": {
                    "CPP_SRC_HOME": "${workspaceFolder}"
                }
            }               
        }
    ]
}
```

## Debugging

To use [debugging](https://code.visualstudio.com/docs/editor/debugging), you need to specify the binary and the
arguments in a json file. You can use `preLaunchTask` to specify the user tasks that should be executed before the start
(e.g. building project).

Here is an example of the `launch.json` file. Please replace `YOUR_PLATFORM` in the `program` path with your 
platform specific value (`linux-x86_64`, `darwin-x86_64`, etc):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "data_frame_analyzer",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/distribution/platform/YOUR_PLATFORM/bin/data_frame_analyzer",
            "args": ["--input", "OnlineNewsPopularity_small.csv",
                "--config", "es_regression_configuration_small.json", 
                "--output", "output.txt"],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "Build ml-cpp bin"
        },
        {
            "name": "api unit test",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/lib/api/unittest/ml_test",
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/lib/api/unittest",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "Build ml-cpp api tests"
        },
        {
            "name": "maths unit test",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/lib/maths/unittest/ml_test",
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/lib/maths/unittest",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "Build ml-cpp maths tests"
        }
    ]
}
```

To run a specific unit test add the test class/test name in the `args` field:
```
            "args": [
                "--run_test=CBoostedTreeTest/testPiecewiseConstant"
            ],
```



## Limitations

* `clangd` may not index all references correctly and so refactor-rename may not work correctly across multiple files so
  you need to use find/replace.
