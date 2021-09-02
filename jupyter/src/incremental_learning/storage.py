# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
from pathlib import Path
from typing import Union

from google.cloud import storage

from .config import bucket_name, configs_dir, datasets_dir, jobs_dir, logger


def download_dataset(dataset_name):
    if dataset_exists(dataset_name):
        return True
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    remote_config_path = 'configs/{}.json'.format(dataset_name)
    remote_dataset_path = 'datasets/{}.csv'.format(dataset_name)
    config_blob = bucket.blob(remote_config_path)
    dataset_blob = bucket.blob(remote_dataset_path)

    result = True

    if not config_blob.exists():
        logger.error("File {} does not exist in the Google storage bucket.".format(
            remote_config_path))
        result = False
    else:
        local_config_path = str(configs_dir / "{}.json".format(dataset_name))
        logger.info("Downloading {} from the Google storage bucket to {}.".format(
            remote_config_path, local_config_path))
        config_blob.download_to_filename(local_config_path)

    if not dataset_blob.exists():
        logger.error("File {} does not exist in the Google storage bucket.".format(
            remote_dataset_path))
        result = False
    else:
        local_dataset_path = str(datasets_dir / "{}.csv".format(dataset_name))
        logger.info("Retrieving {} from the Google storage bucket to {}.".format(
            remote_dataset_path, local_dataset_path))
        dataset_blob.download_to_filename(local_dataset_path)

    return result


def dataset_exists(dataset_name):
    local_config_path = configs_dir / "{}.json".format(dataset_name)
    local_dataset_path = datasets_dir / "{}.csv".format(dataset_name)
    if not local_config_path.exists():
        logger.info("File {} does not exist.".format(local_config_path))
    if not local_dataset_path.exists():
        logger.info("File {} does not exist.".format(local_dataset_path))
    return local_config_path.exists() and local_dataset_path.exists()


def download_job(job_name) -> Union[None, Path]:
    """
    Downloads the job file with the give name form Google storage.

    Args:
        job_name (str): job name

    Returns:
        Local path to the downloaded job if download was successful 
        and None otherwise.
    """
    if job_exists(job_name):
        return True
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    remote_job_path = 'jobs/{}'.format(job_name)
    job_blob = bucket.blob(remote_job_path)

    result = None

    if not job_blob.exists():
        logger.error("File {} does not exist in the Google storage bucket.".format(
            remote_job_path))
        result = None
    else:
        local_job_path = jobs_dir / "{}".format(job_name)
        logger.info("Downloading {} from the Google storage bucket to {}.".format(
            remote_job_path, local_job_path))
        job_blob.download_to_filename(str(local_job_path))
        result = local_job_path

    return result


def job_exists(job_name):
    local_job_path = jobs_dir / "{}.json".format(job_name)
    if not local_job_path.exists():
        logger.warning("File {} does not exist.".format(local_job_path))
        return False
    return True
