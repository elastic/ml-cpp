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

#ifndef INCLUDED_ml_model_CAnnotation_h
#define INCLUDED_ml_model_CAnnotation_h

#include <core/CoreTypes.h>

#include <model/ImportExport.h>

#include <string>

namespace ml {
namespace model {

//! \brief Data necessary to create an annotation
class MODEL_EXPORT CAnnotation {
public:
    //! The values of this enum must correspond to a subset of the values of the
    //! Event enum of org.elasticsearch.xpack.core.ml.annotations.Annotation in
    //! the Java code.
    enum EEvent { E_ModelChange = 0, E_CategorizationStatusChange = 1 };

    //! Detector index will not be written to the output if this value is used.
    static const int DETECTOR_INDEX_NOT_APPLICABLE = -1;

public:
    CAnnotation() = default;
    CAnnotation(core_t::TTime time,
                EEvent event,
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
    const std::string& event() const;
    int detectorIndex() const;
    const std::string& partitionFieldName() const;
    const std::string& partitionFieldValue() const;
    const std::string& overFieldName() const;
    const std::string& overFieldValue() const;
    const std::string& byFieldName() const;
    const std::string& byFieldValue() const;

private:
    core_t::TTime m_Time = 0;
    EEvent m_Event = E_ModelChange;
    std::string m_Annotation;
    int m_DetectorIndex = DETECTOR_INDEX_NOT_APPLICABLE;
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
