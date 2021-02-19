# Run jupyter

- [Run jupyter](#run-jupyter)
  - [Running locally for development](#running-locally-for-development)
  - [Running inside docker container](#running-inside-docker-container)
    - [Build and push docker container](#build-and-push-docker-container)
    - [Running docker container locally](#running-docker-container-locally)
      - [Authorize docker to push to the google image registry](#authorize-docker-to-push-to-the-google-image-registry)
      - [Pushing docker container to the google registry](#pushing-docker-container-to-the-google-registry)
    - [Create an GCP instance from your docker container](#create-an-gcp-instance-from-your-docker-container)
    - [Working with the GCP dataset bucket](#working-with-the-gcp-dataset-bucket)
      - [Install `gcsfuse`](#install-gcsfuse)
      - [Working with the bucket](#working-with-the-bucket)
      - [Copy data without mounting](#copy-data-without-mounting)

## Running locally for development

## Running inside docker container

### Build and push docker container

Since the jupyter notebooks reside in the same repository as the C++ files, `docker` will try to re-build the `data_frame_analyzer` even if you only change the `ipynb` files.

To remedy this and to accelerate the building process, we use `ccache` within docker container when building `ml-cpp`. To use this experimental functionality, you need to set `DOCKER_BUILDKIT` environment variable before running `docker build`.

To build the docker container, run the following command from the root of your `ml-cpp` project:

```bash
DOCKER_BUILDKIT=1 docker build -t myjupyter -f ./dev-tools/docker/jupyter-server/Dockerfile .
```

For debugging purposes, it may be useful to build only the C++ part. You can do it like this:

```bash
DOCKER_BUILDKIT=1 docker build --target builder -t myjupyter:builder \
-f ./dev-tools/docker/jupyter-server/Dockerfile .
```

### Running docker container locally

```bash
docker run --privileged -p 9999:9999 -v ~/data:/data myjupyter:latest
```

#### Authorize docker to push to the google image registry

Before pushing the docker container for the first time to the `gcr` registry, you need to authorize
docker to do this.

```bash
gcloud auth configure-docker
```

This will lead you to a web page where you will need to login and confirm the authorization.

#### Pushing docker container to the google registry

First, you need to tag you docker container with a qualified name to push it to the `gcr` registry. I suggest that you use a unique label (instead of `latest`).

```bash
docker tag myjupyter:latest gcr.io/elastic-ml/incremental-learning-jupyter:try42
```

```bash
docker push gcr.io/elastic-ml/valeriy-jupyter:latest
```

Since docker images are layered, if you have change a lot in some of the layers, it may take a bit.

### Create an GCP instance from your docker container

GCP has a predefined set of machines that we can use. For example, you can select one of the following machine types

| Machine type   | Virtual CPUs | Memory |
| -------------- | ------------ | ------ |
| e2-standard-2  | 2            | 8GB    |
| e2-standard-4  | 4            | 16GB   |
| e2-standard-8  | 8            | 32GB   |
| e2-standard-16 | 16           | 64GB   |
| e2-standard-32 | 32           | 128GB  |

E2 machine types offer up to 32 vCPU with up to 128GB memory. This machine type is probably the best cost-optimized option for use.

To create a compute instance using machine type `e2-standard-16` with 200GB local disk space:

```bash
gcloud compute instances create-with-container valeriy-jupyter-mlcpp-large \
--container-image=gcr.io/elastic-ml/valeriy-jupyter:latest --boot-disk-size=200GB \
--container-privileged --tags=https-server,incremental-jupyter-server,allow-ssh \
--machine-type=e2-standard-16
```

Alternatively, you can specify a [custom machine type](https://cloud.google.com/compute/docs/instances/creating-instance-with-custom-machine-type#gcloud). For instance, to create an instance with 16 CPUs and 32 GB memory, you can specify

```bash
--custom-cpu 16 --custom-memory 32GB --custom-vm-type e2
```

### Working with the GCP dataset bucket

#### Install `gcsfuse`

Follow [the instructions](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md) to install `gcsfuse` on your system. If you have problems mounting the dataset bucket, you may need generate the authorization token for  `gcsfuse`:

```bash
gcloud auth application-default login
```

#### Working with the bucket

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
