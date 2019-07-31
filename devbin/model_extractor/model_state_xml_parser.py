#!/usr/bin/env python3
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

#
# Description:
# Example python script demonstrating the steps required to parse a sequence
# of documents containing residual model state as generated by the model_extractor
# executable using the 'XML' output option and printing statistics of interest.
#
# Requirements:
# * Python 3.x
#
# Usage:
# model_state_parser [<filename>|<stdin>]
# Input is expected to be a sequence of documents in the following format separated by a newline and a zero byte
#{"index":{"_id":"job_model_state_1484006460"}}
#{"xml":"<root><residual_model><one-of-n><7.1/><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><gamma><decay_rate>1.666667e-5</decay_rate><offset>1e-1</offset><likelihood_shape>1</likelihood_shape><log_samples_mean>3.33333333333333e-2:7.20978392236746</log_samples_mean><sample_moments>3.33333333333333e-2:1352.60000000149:0</sample_moments><prior_shape>1</prior_shape><prior_rate>0</prior_rate><number_samples>3.333334e-2</number_samples><mean>&lt;unknown&gt;</mean><standard_deviation>&lt;unknown&gt;</standard_deviation></gamma></prior></model><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><log_normal><decay_rate>1.666667e-5</decay_rate><offset>1</offset><gaussian_mean>7.210449</gaussian_mean><gaussian_precision>3.333111e-2</gaussian_precision><gamma_shape>1.016666</gamma_shape><gamma_rate>7.58141964714308e-10</gamma_rate><number_samples>3.333111e-2</number_samples><mean>1352.5</mean><standard_deviation>0.2057971</standard_deviation></log_normal></prior></model><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><normal><decay_rate>1.666667e-5</decay_rate><gaussian_mean>1352.5</gaussian_mean><gaussian_precision>3.333111e-2</gaussian_precision><gamma_shape>1.016666</gamma_shape><gamma_rate>1.38888737104836e-3</gamma_rate><number_samples>3.333111e-2</number_samples><mean>1352</mean><standard_deviation>1.607356</standard_deviation></normal></prior></model><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><poisson><decay_rate>1.666667e-5</decay_rate><offset>0</offset><shape>45.16366</shape><rate>3.333112e-2</rate><number_samples>3.333111e-2</number_samples><mean>1355</mean><standard_deviation>204.9578</standard_deviation></poisson></prior></model><model><weight><log_weight>-1.79755531839993e308</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><multimodal><clusterer><x_means_online_1d><cluster><index>0</index><prior><decay_rate>1.666667e-5</decay_rate><gaussian_mean>1352.5</gaussian_mean><gaussian_precision>3.333334e-2</gaussian_precision><gamma_shape>1.016667</gamma_shape><gamma_rate>1.38888888888889e-3</gamma_rate><number_samples>3.333334e-2</number_samples><mean>1352</mean><standard_deviation>1.607259</standard_deviation></prior><structure><decay_rate>1.666667e-5</decay_rate><space>12</space><category><size>0</size></category><points>1352;3.333334e-2</points></structure></cluster><available_distributions>7</available_distributions><decay_rate>1.666667e-5</decay_rate><history_length>0</history_length><smallest>1352</smallest><largest>1352</largest><weight>1</weight><cluster_fraction>5e-2</cluster_fraction><minimum_cluster_count>12</minimum_cluster_count><winsorisation_confidence_interval>1</winsorisation_confidence_interval><index_generator><index>1</index></index_generator></x_means_online_1d></clusterer><seed_prior><one-of-n><7.1/><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><gamma><decay_rate>1.666667e-5</decay_rate><offset>1e-1</offset><likelihood_shape>1</likelihood_shape><log_samples_mean>0:0</log_samples_mean><sample_moments>0:0:0</sample_moments><prior_shape>1</prior_shape><prior_rate>0</prior_rate><number_samples>0</number_samples><mean>&lt;unknown&gt;</mean><standard_deviation>&lt;unknown&gt;</standard_deviation></gamma></prior></model><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><log_normal><decay_rate>1.666667e-5</decay_rate><offset>1</offset><gaussian_mean>0</gaussian_mean><gaussian_precision>0</gaussian_precision><gamma_shape>1</gamma_shape><gamma_rate>0</gamma_rate><number_samples>0</number_samples><mean>&lt;unknown&gt;</mean><standard_deviation>&lt;unknown&gt;</standard_deviation></log_normal></prior></model><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><normal><decay_rate>1.666667e-5</decay_rate><gaussian_mean>0</gaussian_mean><gaussian_precision>0</gaussian_precision><gamma_shape>1</gamma_shape><gamma_rate>0</gamma_rate><number_samples>0</number_samples><mean>&lt;unknown&gt;</mean><standard_deviation>&lt;unknown&gt;</standard_deviation></normal></prior></model><sample_moments>0:0:0</sample_moments><decay_rate>1.666667e-5</decay_rate><number_samples>0</number_samples></one-of-n></seed_prior><mode><index>0</index><prior><one-of-n><7.1/><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><gamma><decay_rate>1.666667e-5</decay_rate><offset>1e-1</offset><likelihood_shape>1</likelihood_shape><log_samples_mean>3.33333333333333e-2:7.20978392236746</log_samples_mean><sample_moments>3.33333333333333e-2:1352.60000000149:0</sample_moments><prior_shape>1</prior_shape><prior_rate>0</prior_rate><number_samples>3.333334e-2</number_samples><mean>&lt;unknown&gt;</mean><standard_deviation>&lt;unknown&gt;</standard_deviation></gamma></prior></model><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><log_normal><decay_rate>1.666667e-5</decay_rate><offset>1</offset><gaussian_mean>7.210449</gaussian_mean><gaussian_precision>3.333334e-2</gaussian_precision><gamma_shape>1.016667</gamma_shape><gamma_rate>7.58142793247006e-10</gamma_rate><number_samples>3.333334e-2</number_samples><mean>1352.5</mean><standard_deviation>0.2057905</standard_deviation></log_normal></prior></model><model><weight><log_weight>0</log_weight><long_term_log_weight>0</long_term_log_weight></weight><prior><normal><decay_rate>1.666667e-5</decay_rate><gaussian_mean>1352.5</gaussian_mean><gaussian_precision>3.333334e-2</gaussian_precision><gamma_shape>1.016667</gamma_shape><gamma_rate>1.38888888888889e-3</gamma_rate><number_samples>3.333334e-2</number_samples><mean>1352</mean><standard_deviation>1.607259</standard_deviation></normal></prior></model><sample_moments>3.333334e-2:1352:0</sample_moments><decay_rate>1.666667e-5</decay_rate><number_samples>3.333334e-2</number_samples></one-of-n></prior></mode><decay_rate>1.666667e-5</decay_rate><number_samples>3.333334e-2</number_samples></multimodal></prior></model><sample_moments>3.333111e-2:1352:0</sample_moments><decay_rate>1.666667e-5</decay_rate><number_samples>3.333111e-2</number_samples></one-of-n></residual_model></root>"}


import argparse
import json
import sys
import xml.etree.ElementTree as ET
from math import exp

def parse_model_state_xml(xml_string):
    root = ET.fromstring(xml_string)
    for model in root.findall('./residual_model/one-of-n/model'):
        log_weight = float(model.find('./weight/log_weight').text)
        prior = model.find('./prior/')
        name = prior.tag
        if name == 'multimodal':
            continue
        meanStr = prior.find('./mean').text
        sdStr = prior.find('./standard_deviation').text
        if meanStr != '<unknown>' and sdStr != '<unknown>':
            mean = float(meanStr)
            sd = float(sdStr)
            print("\t{name}: weight = {weight:f}, mean = {mean:f}, sd = {sd:f}"
                    .format(name=name, weight=exp(log_weight), mean=mean, sd=sd))
    return

def parse_model_state_json(json_string):
    try:
        obj = json.loads(json_string)
        if 'index'  in obj:
            print("Residual data for index id {}".format(obj['index']['_id']))
        elif 'xml' in obj:
            xml_string = obj['xml']
            parse_model_state_xml(xml_string.replace('<7.1', '<v7.1'))
        else:
            pass
    except:
        sys.exit("Error: Cannot parse JSON document. Encountered " + str(sys.exc_info()[0]))

    return

if __name__ == '__main__':

    data=''
    parser = argparse.ArgumentParser(description="Parse a sequence of model state documents in the model_extractor\
            \"XML\" output format, from file or stdin")
    parser.add_argument("infile", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file")

    args = parser.parse_args()
    data=args.infile.read()

    # Input is expected to be in standard ES Ml format with each document separated by a newline followed by a zero byte
    lines = [buf.split('\x00') for buf in data.splitlines()]
    [parse_model_state_json(line[0]) for line in lines if line[0] != '']
