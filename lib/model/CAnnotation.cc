/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <model/CAnnotation.h>

namespace ml {
namespace model {
namespace {
// These strings must correspond exactly to lowercased values of the Event enum
// of org.elasticsearch.xpack.core.ml.annotations.Annotation in the Java code.
const std::string EVENT_MODEL_CHANGE{"model_change"};
const std::string EVENT_CATEGORIZATION_STATUS_CHANGE{"categorization_status_change"};
}

CAnnotation::CAnnotation(core_t::TTime time,
                         EEvent event,
                         const std::string& annotation,
                         int detectorIndex,
                         const std::string& partitionFieldName,
                         const std::string& partitionFieldValue,
                         const std::string& overFieldName,
                         const std::string& overFieldValue,
                         const std::string& byFieldName,
                         const std::string& byFieldValue)
    : m_Time{time}, m_Event{event}, m_Annotation{annotation},
      m_DetectorIndex{detectorIndex}, m_PartitionFieldName{partitionFieldName},
      m_PartitionFieldValue{partitionFieldValue}, m_OverFieldName{overFieldName},
      m_OverFieldValue{overFieldValue}, m_ByFieldName{byFieldName}, m_ByFieldValue{byFieldValue} {
}

core_t::TTime CAnnotation::time() const {
    return m_Time;
}

const std::string& CAnnotation::annotation() const {
    return m_Annotation;
}

const std::string& CAnnotation::event() const {
    switch (m_Event) {
    case E_ModelChange:
        return EVENT_MODEL_CHANGE;
    case E_CategorizationStatusChange:
        return EVENT_CATEGORIZATION_STATUS_CHANGE;
    }
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
