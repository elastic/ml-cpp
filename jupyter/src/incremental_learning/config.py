import configparser
import logging
import platform
from sys import exit

from pathlib2 import Path

root_dir = Path(__file__).parent.parent.parent.absolute()
data_dir = root_dir / 'data'
datasets_dir = data_dir / 'datasets'
configs_dir = data_dir / 'configs'
dfa_path = ''

# I assume, your host OS is not CentOS
cloud = (platform.system() == 'Linux') and (platform.dist()[0] == 'centos')
if cloud:
    search_location = Path('/ml-cpp')
    dfa_path = Path('/ml-cpp/bin/data_frame_analyzer')
else:
    search_location = Path(__file__).parent.parent.parent.parent
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

# Read config file
config = configparser.ConfigParser()
config_file = root_dir / 'config.ini'
if not config_file.exists():
    logger.error("Configuration file {} not found.".format(config_file))
    exit(1)
config.read(root_dir / 'config.ini')

# Elasticsearch deployment configuration
if (not config['cloud']['cloud_id']) or (not config['cloud']['user']) or (not config['cloud']['password']):
    logger.error("Cloud configuration is missing or incomplete.")
    exit(1)
es_cloud_id = config['cloud']['cloud_id']
es_user = config['cloud']['user']
es_password = config['cloud']['password']

if config['logging']['level']:
    logger.setLevel(config['logging']['level'])

# Google cloud storage bucket with datasets
if not config['storage']['bucket_name']:
    logger.error("Storage bucket configuration is missing")
    exit(1)
bucket_name = config['storage']['bucket_name']
