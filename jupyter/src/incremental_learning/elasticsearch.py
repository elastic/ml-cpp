import configparser
import json
from pathlib2 import Path
import shutil

from elasticsearch import Elasticsearch

from .config import root_dir

def push2es(data_path, name, prefix='experiment-'):

    config = configparser.ConfigParser()

    config.read(root_dir / 'config.ini')
    es = Elasticsearch(cloud_id=config['cloud']['cloud_id'],
                    http_auth=(config['cloud']['user'], config['cloud']['password']))

    for data_path in data_path.iterdir():
        if not (data_path/'config.json').is_file():
            print('{} ignored'.format(data_path))
            continue

        doc={}
        with open(data_path/'config.json') as fp:
            config=json.load(fp)
            doc['config']=config

        with open(data_path/'metrics.json') as fp:
            metrics=json.load(fp)
            doc['metrics']=metrics

        with open(data_path/'run.json') as fp:
            run=json.load(fp)
            doc['run']=run
            artifacts=run['artifacts']
            run['artifacts']=[]
            for artifact in artifacts:
                with open(data_path/artifact) as fp:
                    run['artifacts'].append({'name': artifact, 'value': fp.read()})

        with open(data_path/'cout.txt') as fp:
            cout=fp.read()
            doc['cout']=cout

        res=es.index(index=prefix+name, body=doc)
        print('{} {}'.format(data_path, res['result']))
