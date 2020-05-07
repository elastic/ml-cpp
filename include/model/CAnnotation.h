/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CAnnotation_h
#define INCLUDED_ml_model_CAnnotation_h

#include <model/ImportExport.h>

#include <model/ModelTypes.h>

#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

namespace ml {
namespace model {

//! \brief Data necessary to create an annotation
class MODEL_EXPORT CAnnotation {
public:
    CAnnotation();
    CAnnotation(core_t::TTime time,
                const std::string& annotation,
                int detectorIndex,
                const std::string& partitionFieldName,
                const std::string& partitionFieldValue,
                const std::string& overFieldName,
                const std::string& overFieldValue,
                const std::string& byFieldName,
                const std::string& byFieldValue);
    core_t::TTime time() const;
    const std::string& annotation() const;
    int detectorIndex() const;
    const std::string& partitionFieldName() const;
    const std::string& partitionFieldValue() const;
    const std::string& overFieldName() const;
    const std::string& overFieldValue() const;
    const std::string& byFieldName() const;
    const std::string& byFieldValue() const;

private:
    core_t::TTime m_Time;
    std::string m_Annotation;
    int m_DetectorIndex;
    std::string m_PartitionFieldName;
    std::string m_PartitionFieldValue;
    std::string m_OverFieldName;
    std::string m_OverFieldValue;
    std::string m_ByFieldName;
    std::string m_ByFieldValue;
};
}
}

#endif // INCLUDED_ml_model_CAnnotation_h
