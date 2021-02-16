# Requirements 

## Install `gcsfuse`
https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md
gcloud auth application-default login
gcsfuse valeriy-data-bucket ~/data

docker build -t myjupyter .

docker run --privileged -p 9999:9999 -v ~/data:/data myjupyter:latest

docker tag myjupyter:latest gcr.io/elastic-ml/valeriy-jupyter:latest

gcloud auth configure-docker

docker push gcr.io/elastic-ml/valeriy-jupyter:latest

gcloud compute instances create-with-container valeriy-jupyter-mlcpp --container-image=gcr.io/elastic-ml/valeriy-jupyter:latest --boot-disk-size=200GB --container-privileged --tags=https-server,valeriy-jupyter-server