# Machine Learning for the Elastic Stack

<https://www.elastic.co/products/stack/machine-learning>

The ml-cpp repo is a part of Machine Learning for the Elastic Stack, which is
available with either a trial or platinum license for the
[Elastic Stack](https://www.elastic.co/products).

This repo only contains the the C++ code that implements the core analytics for
machine learning.

Code for integrating into the Elasticsearch and source for its documentation can
be found in the main
[elasticsearch repo](https://github.com/elastic/elasticsearch).

## Elastic License Functionality

Most files in this repository are subject to the Elastic License. The full
license can be found in [LICENSE.txt](LICENSE.txt). Usage requires a
[subscription](https://www.elastic.co/subscriptions). Files that are not
subject to the Elastic License are in the [3rd_party](3rd_party) and
[build-setup](build-setup) directories.

## Getting Started

Before starting with Machine Learning, it's a good idea to get some experience
with the rest of the
[Elastic Stack](https://www.elastic.co/guide/en/elastic-stack/current/index.html)
first.

To get started with Machine Learning please have a look at
<https://www.elastic.co/guide/en/elastic-stack-overview/current/ml-getting-started.html>.

Full documentation of Machine Learning can be found at
<https://www.elastic.co/guide/en/elastic-stack-overview/current/xpack-ml.html>.

## Questions/Bug Reports/Help

We are happy to help and to make sure your questions can be answered by the
right people, please follow the guidelines below:

* If you have a general question about functionality please use our
  [discuss](https://discuss.elastic.co/tags/c/elasticsearch/machine-learning)
  forums.
* If you have a support contract please use your dedicated support channel.
* For questions regarding subscriptions please
  [contact](https://www.elastic.co/subscriptions#request-info) us.
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

