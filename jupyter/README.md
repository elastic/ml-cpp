# Rapid Prototyping with Python <!-- omit in toc -->

- [Development Installation](#development-installation)
  - [Set up your environment](#set-up-your-environment)
  - [Complete the configuration](#complete-the-configuration)
  - [Check that everything works](#check-that-everything-works)
- [Working with Docker and Google Cloud Platform](#working-with-docker-and-google-cloud-platform)
  - [Running Docker locally](#running-docker-locally)
  - [Push Docker images to the Google Image Registry](#push-docker-images-to-the-google-image-registry)
  - [Create a GCP instance from your Docker image](#create-a-gcp-instance-from-your-docker-image)
- [Directory structure](#directory-structure)
- [FAQ](#faq)
  - [How to build parts of the Docker image separately?](#how-to-build-parts-of-the-docker-image-separately)
  - [What to do if `docker build` fails with the segfault message?](#what-to-do-if-docker-build-fails-with-the-segfault-message)
  - [How to run large jobs on the Google Cloud Platform?](#how-to-run-large-jobs-on-the-google-cloud-platform)
  - [How to upload data to the GCP buckets?](#how-to-upload-data-to-the-gcp-buckets)
  - [How to generate the service account key json file?](#how-to-generate-the-service-account-key-json-file)
  - [How to authorize `docker` to push to the Google Image Registry?](#how-to-authorize-docker-to-push-to-the-google-image-registry)
  - [How to choose the GCP machine configuration?](#how-to-choose-the-gcp-machine-configuration)

## Development Installation

### Set up your environment

1. Install [Docker](https://www.docker.com/) following [their installation guide](https://docs.docker.com/get-docker/).

2. Navigate to `ml-cpp/jupyter` and set up a virtual environment called `env`

    ```bash
    python3 -m venv env
    ```

3. Activate the virtual environment:

    ```bash
    source env/bin/activate
    ```

4. Install the required dependencies:

    ```bash
    cmake .
    ```

### Complete the configuration

1. Specify `cloud_id`, `user`, and `password` in the `[cloud]` section of the `config.ini` file. Note that we commit `template_config.ini` to avoid accidentally committing real credentials and this should be copied and updated to create the `config.ini` file.

2. Generate the service account json file by following using [instructions below](#how-to-generate-the-service-account-key-json-file).

### Check that everything works

1. Activate the virtual environment activated:

    ```bash
    source env/bin/activate
    ```

2. Run Python tests:

    ```bash
    cmake --build . --target src_tests
    ```

3. Run Jupyter notebook smoke tests:

    ```bash
    cmake --build . --target notebook_tests
    ```

4. Build the Docker image:

    ```bash
    IMAGE_NAME=myjupyter cmake --build . --target build_docker
    ```

> *In the [`Dockerfile`](docker/Dockerfile), we fetch the base image from > `docker.elastic.co`. To this end, you need to be
> authorized and authenticated to use `docker.elastic.co` (e.g., your > GitHub account should be a part of *elastic* org).
> Alternatively, you can create the base image yourself using this
> [`Dockerfile`](../dev-tools/docker/linux_image/Dockerfile), albeit it > will take a while.*

## Working with Docker and Google Cloud Platform

### Running Docker locally

> *Follow [these instructions](#how-to-generate-the-service-account-key-json-file) to generate the service account `json` file if you don't have one.*

1. Run the docker container and forward port 9999. To access the data from the Google Cloud Storage, you'll need to additionally pass the key file of the service account and configure the environment variable `GOOGLE_APPLICATION_CREDENTIALS`:

    ```bash
    docker run -p 9999:9999 -v /path/to/gcp-sa.json:/tmp/keys/gcp-sa.json:ro \
    -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gcp-sa.json myjupyter:latest
    ```

2. Navigate to [https://localhost:9999](https://localhost:9999). Since it is HTTPS and we are using self-signed
certificates, you need to confirm the security exception.

### Push Docker images to the Google Image Registry

>*To push a Docker image to the Google Image Registry, you need to get authorization by following [these instructions](#how-to-authorize-docker-to-push-to-the-google-image-registry).*

1. Tag your Docker image with a qualified name to push it to the `gcr` registry. Use a descriptive unique label (instead of `tag`).

    ```bash
    docker tag myjupyter:latest gcr.io/elastic-ml/incremental-learning-jupyter:tag
    ```

2. Push the Docker image to the registry (this may take a while):

    ```bash
    docker push gcr.io/elastic-ml/incremental-learning-jupyter:tag
    ```

### Create a GCP instance from your Docker image

1. Create a compute instance using machine type `e2-standard-16` with 200GB local disk space (or use another machine configuration as [descibed here](#how-to-choose-the-gcp-machine-configuration)):

    ```bash
    gcloud compute instances create-with-container jupyter-mlcpp-large \
    --container-image=gcr.io/elastic-ml/incremental-learning-jupyter:tag --boot-disk-size=200GB \
    --container-privileged --tags=https-server,incremental-learning-jupyter-server,allow-ssh \
    --machine-type=e2-standard-16
    ```

    Once the instance is created, you receive the external IP address of the instance. `tag` should match the label you chose for the Docker image which you pushed to the registry.

2. Navigate to `https://<external_ip>:9999`. Since it is HTTPS and we are using self-signed certificates, you need to confirm the security exception. Note that this can take a short period to become available after creating.

3. If you don't need it anymore, delete the instance:

    ```bash
    gcloud compute instances delete jupyter-mlcpp-large
    ```

## Directory structure

- `data/` datasets and config files. For testing purposes, we have some small example datasets.
- `docker/` scripts and configurations for creating Docker containers.
- `notebooks/` contains `jupyter` notebooks for different projects. Currently, only `incremental_learning` project is available. The project notebooks are used to test functionality and show different behavior. Since they are subject to smoke tests, they should run reasonably fast. If you are working on long-running notebooks, you can put them into `prototypes` directory.
- `scripts/` collection of python and shell scripts, e.g., experiment drivers.
- `src/` python packages containing supporting code. Currently, only `incremental_learning` package is available.
- `tests/` pytest unit tests for python packages.
- `CMakeLists.txt` for automated tasks. Run `cmake --build . --target help` to see which tasks are available.

## FAQ

### How to build parts of the Docker image separately?

For debugging purposes, it may be useful to build only the C++ part. You can do it like this:

```bash
DOCKER_BUILDKIT=1 docker build --target builder -t myjupyter:builder \
-f ./jupyter/docker/Dockerfile .
```

### What to do if `docker build` fails with the segfault message?

If during the build process you face errors due to out-of-memory, consider increasing container memory using `--memory`
parameter (e.g., `--memory=4g` for 4 GB of memory in the container).

### How to run large jobs on the Google Cloud Platform?

All calls to `data_frame_analyzer` are performed in `tmux`. This ensures that even you lose the connection or close your
browser, you can go back to your Jupyter notebook and open the running notebook (e.g., from the tab `Running`). If it is
still in the call to `job.wait_job_complete()`, you can interrupt the kernel and re-run the last cell. The output in the
cell is updated again. The state of the variables should also be unchanged.

### How to upload data to the GCP buckets?

To mount the dataset bucket locally to `~/data` (the directory should already exist):

```bash
gcsfuse ml-incremental-learning-datasets ~/data
```

To unmount the bucket:

```bash
fusermount -u ~/data
```

You may need to add the parameter `-z` to force the unmounting. (Note that `gcsfuse` does not support the latest build
of FUSE for macOS; see [this](https://github.com/GoogleCloudPlatform/gcsfuse/issues/514) for more details.) 

To copy `dataset.csv` to the bucket without mounting it:

```bash
gsutil cp dataset.csv gs://ml-incremental-learning-datasets/datasets
```

### How to generate the service account key json file?

If you are authorized, you can generate the json file with service account information using the command below.
Don't forget to substitute `/path/to/gcp-sa.json` with your path!

```bash
gcloud iam service-accounts keys create /path/to/gcp-sa.json \
--iam-account=incremental-training-reader@elastic-ml.iam.gserviceaccount.com
```

### How to authorize `docker` to push to the Google Image Registry?

Before pushing the docker container for the first time to the `gcr` registry, you need to authorize
docker to do this.

```bash
gcloud auth configure-docker
```

This command leads you to a web page where you need to log in and confirm the authorization.

### How to choose the GCP machine configuration?

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

Alternatively, you can specify a
[custom machine type](https://cloud.google.com/compute/docs/instances/creating-instance-with-custom-machine-type#gcloud).
For instance, to create an instance with 16 CPUs and 32 GB memory, you can specify

```bash
--custom-cpu 16 --custom-memory 32GB --custom-vm-type e2
```

instead of the `--machine-type` parameter.
