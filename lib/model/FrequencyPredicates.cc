/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/FrequencyPredicates.h>

namespace ml {
namespace model {

CPersonFrequencyGreaterThan::CPersonFrequencyGreaterThan(const CAnomalyDetectorModel& model, double threshold)
    : m_Model(&model), m_Threshold(threshold) {
}

CAttributeFrequencyGreaterThan::CAttributeFrequencyGreaterThan(const CAnomalyDetectorModel& model, double threshold)
    : m_Model(&model), m_Threshold(threshold) {
}
}
}
