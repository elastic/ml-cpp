import json
import shutil

from pathlib2 import Path

from elasticsearch import Elasticsearch

from .config import es_cloud_id, es_password, es_user, logger, root_dir


def push2es(data_path, name, prefix='experiment-'):
    es = Elasticsearch(cloud_id=es_cloud_id,
                       http_auth=(es_user, es_password))

    for data_path in data_path.iterdir():
        if not (data_path/'config.json').is_file():
            print('{} ignored'.format(data_path))
            continue

        doc = {}
        with open(data_path/'config.json') as fp:
            config = json.load(fp)
            doc['config'] = config

        with open(data_path/'metrics.json') as fp:
            metrics = json.load(fp)
            doc['metrics'] = metrics

        with open(data_path/'run.json') as fp:
            run = json.load(fp)
            doc['run'] = run
            artifacts = run['artifacts']
            run['artifacts'] = []
            for artifact in artifacts:
                with open(data_path/artifact) as fp:
                    run['artifacts'].append(
                        {'name': artifact, 'value': fp.read()})

        with open(data_path/'cout.txt') as fp:
            cout = fp.read()
            doc['cout'] = cout

        res = es.index(index=prefix+name, body=doc)
        logger.info('{} {}'.format(data_path, res['result']))
