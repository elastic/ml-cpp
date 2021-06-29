# Evaluation Workbench for Incremental Learning <!-- omit in toc -->

- [Directory structure](#directory-structure)
- [Zero-to-hero launching `jupyter notebook`](#zero-to-hero-launchingjupyter-notebook)
  - [Running locally for development](#running-locally-for-development)
  - [Running inside the docker container](#running-inside-the-docker-container)
    - [Build and push docker container](#build-and-push-docker-container)
    - [Running docker container locally](#running-docker-container-locally)
      - [Authorize docker to push to the google image registry](#authorize-docker-to-push-to-the-google-image-registry)
      - [Pushing docker container to the google registry](#pushing-docker-container-to-the-google-registry)
    - [Create an GCP instance from your docker container](#create-an-gcp-instance-from-your-docker-container)
      - [Using custom machine configuration](#using-custom-machine-configuration)
    - [Working with the GCP dataset bucket](#working-with-the-gcp-dataset-bucket)
      - [Install `gcsfuse`](#install-gcsfuse)
- [Running jupyter notebooks](#running-jupyter-notebooks)
- [Miscellaneous topics](#miscellaneous-topics)
  - [Working with large jobs on GCP](#working-with-large-jobs-on-gcp)
  - [Working with the GCP buckets](#working-with-the-gcp-buckets)
    - [Copy data without mounting](#copy-data-without-mounting)
- [Running tests with `make`](#running-tests-with-make)

## Directory structure

- `data/` datasets and config files. For testing purposes, we have some small example datasets.
- `docker/` scripts and configurations for creating Docker containers.
- `notebooks/` contains `jupyter` notebooks for different projects. Currently, only `incremental_learning` project is available. The project notebooks are used to test functionality and show different behaviour. Since they are subject to smoketests, they should run reasonably fast. If you are working on long-running notebooks, you can put them into `prototypes` directory.
- `src/` python packages containing supporting code. Currently, only `incremental_learning` package is available.
- `tests/` pytest unit tests for python packages.
- `Makefile` for automated tasks. Run `make` to see which tasks are available.

## Zero-to-hero launching `jupyter notebook`

### Running locally for development

Set up a local instance of Jupyter using the following instructions

1. Navigate to `ml-cpp/jupyter` and set up a virtual environment called `env`

    ```bash
    python3 -m venv env
    ```

2. Activate it

    ```bash
    source env/bin/activate
    ```

3. Install the required dependencies for your chosen Jupyter notebook

    ```bash
    make **env**
    ```

4. Launch Jupyter

    ```bash
    jupyter notebook
    ```

### Running inside the docker container

In the following, we assume that you have
[`docker` installed successfully running](https://docs.docker.com/get-started/) on your host system. 

Furthermore, in the [`Dockerfile`](docker/Dockerfile) we fetch the base image from `docker.elastic.co`. To this end, you
need to be authorized and authenticated to use `docker.elastic.co` (e.g., your GitHub account should be a part of
*elastic* org). Alternatively, you can create the base image yourself using this
[`Dockerfile`](../dev-tools/docker/linux_image/Dockerfile) albeit it will take a while. You then will need to tag your
image correctly or change the base image name to the one you have created.

#### Build and push docker container

Since the jupyter notebooks reside in the same repository as the C++ files, `docker` will try to re-build the
`data_frame_analyzer` even if you only change the `ipynb` files.

To remedy this and to accelerate the building process, we use `ccache` within the docker container when building `ml-cpp`.
To use this experimental functionality, you need to set `DOCKER_BUILDKIT` environment variable before running
`docker build`.

To build the docker container, run the following command from the root directory of your `ml-cpp` project:

```bash
DOCKER_BUILDKIT=1 docker build -t myjupyter -f ./jupyter/docker/Dockerfile .
```

For debugging purposes, it may be useful to build only the C++ part. You can do it like this:

```bash
DOCKER_BUILDKIT=1 docker build --target builder -t myjupyter:builder \
-f ./jupyter/docker/Dockerfile .
```

If during the build process you face errors due to out-of-memory, consider increasing container memory using `--memory`
parameter (e.g., `--memory=4g` for 4 GB of memory in the container).

#### Running docker container locally

Run the docker container forwarding port 9999.

```bash
docker run -p 9999:9999 myjupyter:latest
```

Now, **navigate to** [https://localhost:9999](https://localhost:9999). Since it is HTTPS and we are using self-signed
certificates, you will need to confirm security exception.

##### Authorize docker to push to the google image registry

Before pushing the docker container for the first time to the `gcr` registry, you need to authorize
docker to do this.

```bash
gcloud auth configure-docker
```

This will lead you to a web page where you will need to login and confirm the authorization.

##### Pushing docker container to the google registry

First, you need to tag you docker container with a qualified name to push it to the `gcr` registry. I suggest that you
use a unique label (instead of `latest`).

```bash
docker tag myjupyter:latest gcr.io/elastic-ml/incremental-learning-jupyter:try42
```

```bash
docker push gcr.io/elastic-ml/incremental-learning-jupyter:try42
```

Since docker images are layered, if you have change a lot in some of the layers, it may take a bit.

#### Create an GCP instance from your docker container

GCP has a predefined set of machines that we can use. For example, you can select one of the following machine types

| Machine type   | Virtual CPUs | Memory |
| -------------- | ------------ | ------ |
| e2-standard-2  | 2            | 8GB    |
| e2-standard-4  | 4            | 16GB   |
| e2-standard-8  | 8            | 32GB   |
| e2-standard-16 | 16           | 64GB   |
| e2-standard-32 | 32           | 128GB  |

E2 machine types offer up to 32 vCPU with up to 128GB memory. This machine type is probably the best cost-optimized
option for use.

To create a compute instance using machine type `e2-standard-16` with 200GB local disk space:

```bash
gcloud compute instances create-with-container jupyter-mlcpp-large \
--container-image=gcr.io/elastic-ml/incremental-learning-jupyter:try42 --boot-disk-size=200GB \
--container-privileged --tags=https-server,incremental-jupyter-server,allow-ssh \
--machine-type=e2-standard-16
```

Once the instance is created, you will receive the external IP address of the instance.

Now, **navigate to** `https://<external_ip>:9999`. Since it is HTTPS and we are using self-signed certificates, you will
need to confirm the security exception.

##### Using custom machine configuration

Alternatively, you can specify a
[custom machine type](https://cloud.google.com/compute/docs/instances/creating-instance-with-custom-machine-type#gcloud).
For instance, to create an instance with 16 CPUs and 32 GB memory, you can specify

```bash
--custom-cpu 16 --custom-memory 32GB --custom-vm-type e2
```

instead of the `--machine-type` parameter.

#### Working with the GCP dataset bucket

##### Install `gcsfuse`

Follow [the instructions](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md) to install
`gcsfuse` on your system. If you have problems mounting the dataset bucket, you may need to generate the authorization
token for `gcsfuse`:

```bash
gcloud auth application-default login
```

## Running jupyter notebooks

If you run jupyter notebook from GCP, you may want to start by running `init.sh` script to mount the
`ml-incremental-learning-datasets` bucket under `/data` and copy the dataset into the instance.

Within your jupyter notebook instance you will find the directory `notebooks` with the following file structure:

```tree
notebooks
├── configs
│   ├── facebook_6000.json
│   └── ...
├── datasets
│   ├── facebook_6000.csv
│   └── ...
├── scenario-1.ipynb
└── ...
```

Please note that these files are copied from this repository into the docker container. This means that once the
container is deleted, the data will be lost! Make sure to `docker cp` or `gsutil cp` the data you wish to keep.

## Miscellaneous topics

### Working with large jobs on GCP

All calls to `data_frame_analyzer` are performed in `tmux`. This ensures that even you lose the connection or close your
browser, you can go back to your jupyter notebook and open the running notebook (e.g. from the tab `Running`). If it is
still in the call to wait_job_complete(job) you can interrupt the kernel and re-run the last cell. The output in the
cell will be updated again. The state of the variables should also be unchanged.

### Working with the GCP buckets

To mount the dataset bucket locally to `~/data` (the directory should already exist):

```bash
gcsfuse ml-incremental-learning-datasets ~/data
```

To unmount the bucket:

```bash
fusermount -u ~/data
```

You may need to add the parameter `-z` to force the unmounting.

#### Copy data without mounting

To copy `dataset.csv` to the bucket:

```bash
gsutil cp dataset.csv gs://ml-incremental-learning-datasets
```

## Running tests with `make`

To see all available `make` targets run `make` from this directory. The output will be something like 

```
all               Run all
env               Install requirements and python packages
bump-minor        Bump minor version of python packages
bump-major        Bump major version of python packages
test              Run unit tests
test-notebooks    Run jupyter notebooks smoketests
help              Print this help
clean-env         Delete environment
clean-build       Delete temporary files and directories
clean-test        Delete temporary files for testing
clean-notebooks   Delete notebook checkpoints
```

To execute python integration tests, run

```bash
make test
```

To execute jupyter notebooks smoke tests, run

```bash
make test-notebooks
```