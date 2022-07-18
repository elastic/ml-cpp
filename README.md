# Machine Learning for the Elastic Stack

<https://www.elastic.co/what-is/elasticsearch-machine-learning>

The ml-cpp repo is a part of Machine Learning for the Elastic Stack, which is
available with either a trial or platinum license for the
[Elastic Stack](https://www.elastic.co/products).

This repo only contains the the C++ code that implements the core analytics for
machine learning.

Code for integrating into Elasticsearch and source for its documentation can be
found in the main
[elasticsearch repo](https://github.com/elastic/elasticsearch).

## Elastic License Functionality

Usage in production requires that you have a license key that permits use of
machine learning features. See [LICENSE.txt](LICENSE.txt) for full information.

## Getting Started

To get started with Machine Learning please have a look at
<https://www.elastic.co/guide/en/machine-learning/current/ml-getting-started.html>.

Full documentation of Machine Learning can be found at
<https://www.elastic.co/guide/en/machine-learning/current/index.html>.

## Questions/Bug Reports/Help

We are happy to help and to make sure your questions can be answered by the
right people, please follow the guidelines below:

* If you have a general question about functionality please use our
  [discuss](https://discuss.elastic.co/tag/elastic-stack-machine-learning)
  forums.
* If you have a support contract please use your dedicated support channel.
* For questions regarding subscriptions please
  [contact](https://www.elastic.co/contact) us.
* For bug reports, pull requests and feature requests specifically for machine
  learning analytics, please use this GitHub repository.

## Contributing

Please have a look at our [contributor guidelines](CONTRIBUTING.md).

## Setting up a build environment

You don't need to specifically build the C++ components for machine learning as,
by default, the elasticsearch build will download pre-compiled C++ artifacts.

Setting up a build environment for ml-cpp native code is complex. If you are
specifically interested in working with the ml-cpp code, then information
regarding setting up a build environment can be found in the
[build-setup](build-setup) directory.

To use CLion with the project, please refer to the ["Using CLion"](build-setup/clion/using_clion.md) tutorial.

## Building

###

If you do choose to build the project from the command line yourself, for all platforms, the following instructions apply:

* From the top level of the project, source the file `set_env.sh` e.g.
```
. ./set_env.sh
```
When building on Windows from the native command shell that command becomes
```
.\set_env.bat
```

* Run `cmake -B cmake-build-relwithdebinfo` to generate the build system under the `cmake-build-relwithdebinfo` directory (the `--config RelWithDebInfo` option may be omitted on Linux and Mac).
* Run `cmake --build cmake-build-relwithdebinfo --config RelWithDebInfo` to build the libraries and the executables for the project (the `--config RelWithDebInfo` option may be omitted on Linux and Mac). This may take some time, to speed up the build you can tell `cmake` to perform a parallel build using the `-j` (jobs) option. e.g.
```
cmake --build cmake-build-relwithdebinfo -j 7
```

* To build and run the unit tests run `cmake --build cmake-build-relwithdebinfo -t test`. Again this can be sped up somewhat by using the `-j` option. e.g.
```
cmake --build cmake-build-relwithdebinfo -t test -j 7
```

## Running

Although the executables are designed to be run from `Elasticsearch` it is possible to run them from the command line. This is particularly useful when attempting to debug issues and you have an input data set sufficient to replicate the error.

The location of the executables differs depending on the platform. 

* MacOS: `build/distribution/platform/darwin-x86_64/controller.app/Contents/MacOS/`
* Linux: `build/distribution/platform/linux-x86_64/bin/`
* Windows: ` build/distribution/platform/windows-x86_64/bin/`

The command line arguments will of course differ depending on which executable is being run but each has the `--help` option e.g. `

```
./build/distribution/platform/linux-x86_64/bin/autodetect --help
Usage: autodetect [options] [<fieldname>+ [by <fieldname>]]
Options::
  --help                      Display this information and exit
  --version                   Display version information and exit
  --limitconfig arg           Optional limit config file
  --modelconfig arg           Optional model config file
  --fieldconfig arg           Optional field config file
  --modelplotconfig arg       Optional model plot config file
  --jobid arg                 ID of the job this process is associated with
  --logProperties arg         Optional logger properties file
  --logPipe arg               Optional log to named pipe
  --bucketspan arg            Optional aggregation bucket span (in seconds) - 
                              default is 300
  --latency arg               Optional maximum delay for out-of-order records 
                              (in seconds) - default is 0
  --summarycountfield arg     Optional field to that contains counts for 
                              pre-summarized input - default is none
  --delimiter arg             Optional delimiter character for delimited data 
                              formats - default is '' (tab separated)
  --lengthEncodedInput        Take input in length encoded binary format - 
                              default is delimited
  --timefield arg             Optional name of the field containing the 
                              timestamp - default is 'time'
  --timeformat arg            Optional format of the date in the time field in 
                              strptime code - default is the epoch time in 
                              seconds
  --quantilesState arg        Optional file to quantiles for normalization
  --deleteStateFiles          If the 'quantilesState' option is used and this 
                              flag is set then delete the model state files 
                              once they have been read
  --input arg                 Optional file to read input from - not present 
                              means read from STDIN
  --inputIsPipe               Specified input file is a named pipe
  --output arg                Optional file to write output to - not present 
                              means write to STDOUT
  --outputIsPipe              Specified output file is a named pipe
  --restore arg               Optional file to restore state from - not present
                              means no state restoration
  --restoreIsPipe             Specified restore file is a named pipe
  --persist arg               Optional file to persist state to - not present 
                              means no state persistence
  --persistIsPipe             Specified persist file is a named pipe
  --persistInterval arg       Optional time interval at which to periodically 
                              persist model state (Mutually exclusive with 
                              bucketPersistInterval)
  --persistInForeground       Persistence occurs in the foreground. Defaults to
                              background persistence.
  --bucketPersistInterval arg Optional number of buckets after which to 
                              periodically persist model state (Mutually 
                              exclusive with persistInterval)
  --maxQuantileInterval arg   Optional interval at which to periodically output
                              quantiles if they have not been output due to an 
                              anomaly - if not specified then quantiles will 
                              only be output following a big anomaly
  --maxAnomalyRecords arg     The maximum number of records to be outputted for
                              each bucket. Defaults to 100, a value 0 removes 
                              the limit.
  --memoryUsage               Log the model memory usage at the end of the job
  --multivariateByFields      Optional flag to enable multi-variate analysis of
                              correlated by fields

```

Other executables exist under the `devbin` directory. These are not built by default. To build these you need to explicitly specify a target. 
```
cmake --build cmake-build-relwithdebinfo -j 7 -t model_extractor
```
The executable is created under the `cmake-build-relwithdebinfo` hierarchy, so to run do
```
./cmake-build-relwithdebinfo/devbin/model_extractor/model_extractor --help
```


