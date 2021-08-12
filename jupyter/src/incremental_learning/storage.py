# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

from google.cloud import storage
from .config import bucket_name, datasets_dir, configs_dir, logger


def download_dataset(dataset_name):
    if dataset_exists(dataset_name):
        return True
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    remote_config_path = 'configs/{}.json'.format(dataset_name)
    remote_dataset_path = 'datasets/{}.csv'.format(dataset_name)
    config_blob = bucket.blob(remote_config_path)
    dataset_blob = bucket.blob(remote_dataset_path)

    if not config_blob.exists():
        logger.error("File {} does not exist in the Google storage bucket.".format(
            remote_config_path))
    if not dataset_blob.exists():
        logger.error("File {} does not exist in the Google storage bucket.".format(
            remote_dataset_path))
    if config_blob.exists() and dataset_blob.exists():
        local_config_path = str(configs_dir / "{}.json".format(dataset_name))
        logger.info("Downloading {} from the Google storage bucket to {}.".format(
            remote_config_path, local_config_path))
        config_blob.download_to_filename(local_config_path)

        local_dataset_path = str(datasets_dir / "{}.csv".format(dataset_name))
        logger.info("Retrieving {} from the Google storage bucket to {}.".format(
            remote_dataset_path, local_dataset_path))
        dataset_blob.download_to_filename(local_dataset_path)
    else:
        return False
    return True


def dataset_exists(dataset_name):
    local_config_path = configs_dir / "{}.json".format(dataset_name)
    local_dataset_path = datasets_dir / "{}.csv".format(dataset_name)
    if not local_config_path.exists():
        logger.warning("File {} does not exist.".format(local_config_path))
    if not local_dataset_path.exists():
        logger.warning("File {} does not exist.".format(local_dataset_path))
    return local_config_path.exists() and local_dataset_path.exists()
