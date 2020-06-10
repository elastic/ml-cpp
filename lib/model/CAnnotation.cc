/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CAnnotation.h>

namespace ml {
namespace model {
namespace {
const std::string EVENT_MODEL_CHANGE{"model_change"};
}

CAnnotation::CAnnotation(core_t::TTime time,
                         const std::string& annotation,
                         int detectorIndex,
                         const std::string& partitionFieldName,
                         const std::string& partitionFieldValue,
                         const std::string& overFieldName,
                         const std::string& overFieldValue,
                         const std::string& byFieldName,
                         const std::string& byFieldValue)
    : m_Time{time}, m_Annotation{annotation}, m_DetectorIndex{detectorIndex},
      m_PartitionFieldName{partitionFieldName}, m_PartitionFieldValue{partitionFieldValue},
      m_OverFieldName{overFieldName}, m_OverFieldValue{overFieldValue},
      m_ByFieldName{byFieldName}, m_ByFieldValue{byFieldValue} {
}

core_t::TTime CAnnotation::time() const {
    return m_Time;
}

const std::string& CAnnotation::annotation() const {
    return m_Annotation;
}

const std::string& CAnnotation::event() const {
    // If we start reporting some other event type, we should return m_Event instead of a constant here.
    return EVENT_MODEL_CHANGE;
}

int CAnnotation::detectorIndex() const {
    return m_DetectorIndex;
}

const std::string& CAnnotation::partitionFieldName() const {
    return m_PartitionFieldName;
}

const std::string& CAnnotation::partitionFieldValue() const {
    return m_PartitionFieldValue;
}

const std::string& CAnnotation::overFieldName() const {
    return m_OverFieldName;
}

const std::string& CAnnotation::overFieldValue() const {
    return m_OverFieldValue;
}

const std::string& CAnnotation::byFieldName() const {
    return m_ByFieldName;
}

const std::string& CAnnotation::byFieldValue() const {
    return m_ByFieldValue;
}
}
}
