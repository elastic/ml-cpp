# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import json
import itertools

from pathlib2 import Path

from elasticsearch import Elasticsearch

from .config import es_cloud_id, es_password, es_user, logger, root_dir


def push2es(data_path, name, prefix='experiment-'):
    es = Elasticsearch(cloud_id=es_cloud_id,
                       http_auth=(es_user, es_password))
    doc = {'sources': []}
    with open(data_path/'config.json') as fp:
        config = json.load(fp)
        doc['config'] = config
    with open(data_path/'run.json') as fp:
        run = json.load(fp)
        doc['run'] = run
        artifacts = run['artifacts']
        run['artifacts'] = []
        for artifact in artifacts:
            with open(data_path/artifact) as fp:
                run['artifacts'].append(
                    {'name': artifact, 'value': fp.read()})
        sources = run['experiment']['sources']
        # unpack nested list
        sources = list(itertools.chain.from_iterable(sources))
        for source in sources:
            source_path = data_path / ('../' + source)
            if source_path.exists():
                with open(source_path) as fs:
                    doc['sources'].append({'key': source, "value": fs.read()})
                logger.debug(
                    "Source {} is added to the document.".format(source))

    with open(data_path/'cout.txt') as fp:
        cout = fp.read()
        doc['cout'] = cout

    experiment_uid = doc['run']['experiment']['name']
    comment = doc['run']['meta']['comment']
    doc['experiment_uid'] = experiment_uid
    res = es.index(index=prefix+name, body=doc)
    logger.info('Indexing experiment info {} {}'.format(
        data_path, res['result']))

    with open(data_path/'metrics.json') as fp:
        metrics = json.load(fp)
        keys = list(metrics.keys())
        if keys:
            steps = metrics[keys[0]]['steps']
            for step in steps:
                document = {'experiment_uid': experiment_uid, 'step': step, 'dataset_name': doc['config']['dataset_name'], 'comment': comment}
                for k in keys:
                    document[k] = metrics[k]['values'][step]
                res = es.index(index=prefix+name+'-metrics', body=document)
                logger.info('Indexing experiment metrics {} {}'.format(
                    data_path, res['result']))
