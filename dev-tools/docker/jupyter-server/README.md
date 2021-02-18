# Requirements 

## Install `gcsfuse`
https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md
gcloud auth application-default login
gcsfuse valeriy-data-bucket ~/data
gsutil cp *.csv gs://valeriy-data-bucket
fusermount -u -z ~/data

# build ml-cpp image (don't forget to comment the last command in the script)
docker_build.sh linux

DOCKER_BUILDKIT=1 docker build --target builder -t myjupyter:builder -f ./dev-tools/docker/jupyter-server/Dockerfile .
docker run -p 9999:9999 myjupyter:latest /root/env/bin/jupyter notebook

docker build -t myjupyter .

docker run --privileged -p 9999:9999 -v ~/data:/data myjupyter:latest

docker tag myjupyter:latest gcr.io/elastic-ml/valeriy-jupyter:latest

gcloud auth configure-docker

docker push gcr.io/elastic-ml/valeriy-jupyter:latest

gcloud compute instances create-with-container valeriy-jupyter-mlcpp --container-image=gcr.io/elastic-ml/valeriy-jupyter:latest --boot-disk-size=200GB --container-privileged --tags=https-server,valeriy-jupyter-server

# create an instance with specified requirements
Machine type 	Virtual CPUs 	Memory
e2-standard-2 	2 	8GB 
e2-standard-4 	4 	16GB
e2-standard-8 	8 	32GB
e2-standard-16 	16 	64GB 
e2-standard-32 	32 	128GB

gcloud compute instances create-with-container valeriy-jupyter-mlcpp-large --container-image=gcr.io/elastic-ml/valeriy-jupyter:latest --boot-disk-size=200GB --container-privileged --tags=https-server,valeriy-jupyter-server,allow-ssh --machine-type=e2-standard-16

# Dealing with large jobs

Watch the progress of the output file:

```bash
watch -n 5 tail -n10 /tmp/tmpoutput
```
