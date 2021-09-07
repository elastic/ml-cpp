# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import configparser
import logging
import platform
from sys import exit

from pathlib import Path

def find_ancestor_dir(path: Path, resource: str):
    '''
    This searches each parent directory of path looking resource until it finds
    it or reaches the root directory. If it didn't find resource it calls exit
    with failure.
    '''
    current = path
    parent = current.parent
    while parent != current:
        if (parent / resource).exists():
            return parent.absolute()
        current = parent
        parent = current.parent
    print('Did not find', resource, 'in parent of', path)
    exit(1)

root_dir = find_ancestor_dir(path=Path(__file__), resource='data/datasets')
data_dir = root_dir / 'data'
datasets_dir = data_dir / 'datasets'
configs_dir = data_dir / 'configs'
jobs_dir = data_dir / 'jobs'
dfa_path = ''

# Assumes your host OS is not CentOS.
cloud = (platform.system() == 'Linux') and (platform.dist()[0] == 'centos')
if cloud:
    search_location = Path('/ml-cpp')
    dfa_path = Path('/ml-cpp/bin/data_frame_analyzer')
else:
    search_location = find_ancestor_dir(path=root_dir, resource='build/distribution')
    runners = list(search_location.glob(
        '**/build/distribution/platform/**/data_frame_analyzer'))
    if len(runners) == 0:
        print("Cannot find data_frame_analyzer binary")
        exit(1)
    dfa_path = runners[0]

logger = logging.getLogger(__package__)
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Read config file.
config = configparser.ConfigParser()
config_file = root_dir / 'config.ini'
if not config_file.exists():
    logger.error("Configuration file {} not found.".format(config_file))
    exit(1)
if len(config.read(root_dir / 'config.ini')) == 0:
    logger.error("Failed to read configuration file {}.".format(config_file))
    exit(1)
    
es_cloud_id = ""
es_user = ""
es_password = ""

# Elasticsearch deployment configuration.
if (not config['cloud']['cloud_id']) or (not config['cloud']['user']) or (not config['cloud']['password']):
    logger.error("Cloud configuration is missing or incomplete. Some functionality will be broken")
else:
    es_cloud_id = config['cloud']['cloud_id']
    es_user = config['cloud']['user']
    es_password = config['cloud']['password']

if config['logging']['level']:
    logger.setLevel(config['logging']['level'])

# Google cloud storage bucket with datasets.
if not config['storage']['bucket_name']:
    logger.error("Storage bucket configuration is missing")
    exit(1)
bucket_name = config['storage']['bucket_name']
