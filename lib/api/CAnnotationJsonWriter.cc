/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CAnnotationJsonWriter.h>

#include <core/CTimeUtils.h>

#include <model/CAnnotation.h>

namespace ml {
namespace api {
namespace {
// JSON field names
const std::string ANNOTATION_RESULT_TYPE{"annotation"};
const std::string TIMESTAMP{"timestamp"};
const std::string END_TIMESTAMP{"end_timestamp"};
const std::string ANNOTATION{"annotation"};
const std::string CREATE_TIME{"create_time"};
const std::string CREATE_USERNAME{"create_username"};
const std::string MODIFIED_TIME{"modified_time"};
const std::string MODIFIED_USERNAME{"modified_username"};
const std::string TYPE{"type"};
const std::string EVENT{"event"};
const std::string JOB_ID{"job_id"};
const std::string DETECTOR_INDEX{"detector_index"};
const std::string PARTITION_FIELD_NAME{"partition_field_name"};
const std::string PARTITION_FIELD_VALUE{"partition_field_value"};
const std::string OVER_FIELD_NAME{"over_field_name"};
const std::string OVER_FIELD_VALUE{"over_field_value"};
const std::string BY_FIELD_NAME{"by_field_name"};
const std::string BY_FIELD_VALUE{"by_field_value"};

// Type can be annotation or comment, but from the C++ always annotation
const std::string ANNOTATION_TYPE{"annotation"};
// Internal user name
const std::string XPACK_USER{"_xpack"};
}

CAnnotationJsonWriter::CAnnotationJsonWriter(core::CJsonOutputStreamWrapper& outStream)
    : m_Writer{outStream} {
}

void CAnnotationJsonWriter::writeResult(const std::string& jobId,
                                        const model::CAnnotation& annotation) {

    rapidjson::Value obj{m_Writer.makeObject()};
    this->populateAnnotationObject(jobId, annotation, obj);

    rapidjson::Value wrapper{m_Writer.makeObject()};
    m_Writer.addMember(ANNOTATION_RESULT_TYPE, obj, wrapper);
    m_Writer.write(wrapper);
    m_Writer.Flush();
}

void CAnnotationJsonWriter::populateAnnotationObject(const std::string& jobId,
                                                     const model::CAnnotation& annotation,
                                                     rapidjson::Value& obj) {

    // There is no need to copy the strings, as this is a private method and the
    // rapidjson::Value it's populating will have a shorter lifetime than the
    // CAnnotation object the string references rely on.
    m_Writer.addStringFieldReferenceToObj(JOB_ID, jobId, obj);
    m_Writer.addStringFieldReferenceToObj(ANNOTATION, annotation.annotation(), obj, true);
    m_Writer.addStringFieldReferenceToObj(EVENT, annotation.event(), obj);

    // In the JSON the time is in Java format - milliseconds since the epoch
    m_Writer.addTimeFieldToObj(TIMESTAMP, annotation.time(), obj);
    m_Writer.addTimeFieldToObj(END_TIMESTAMP, annotation.time(), obj);

    std::int64_t currentTime{core::CTimeUtils::nowMs()};
    m_Writer.addIntFieldToObj(CREATE_TIME, currentTime, obj);
    m_Writer.addIntFieldToObj(MODIFIED_TIME, currentTime, obj);
    m_Writer.addStringFieldReferenceToObj(CREATE_USERNAME, XPACK_USER, obj);
    m_Writer.addStringFieldReferenceToObj(MODIFIED_USERNAME, XPACK_USER, obj);
    m_Writer.addStringFieldReferenceToObj(TYPE, ANNOTATION_TYPE, obj);

    if (annotation.detectorIndex() != model::CAnnotation::DETECTOR_INDEX_NOT_APPLICABLE) {
        m_Writer.addIntFieldToObj(DETECTOR_INDEX, annotation.detectorIndex(), obj);
    }
    if (annotation.partitionFieldName().empty() == false) {
        m_Writer.addStringFieldReferenceToObj(PARTITION_FIELD_NAME,
                                              annotation.partitionFieldName(), obj);
        m_Writer.addStringFieldReferenceToObj(
            PARTITION_FIELD_VALUE, annotation.partitionFieldValue(), obj, true);
    }
    if (annotation.overFieldName().empty() == false) {
        m_Writer.addStringFieldReferenceToObj(OVER_FIELD_NAME,
                                              annotation.overFieldName(), obj);
        m_Writer.addStringFieldReferenceToObj(
            OVER_FIELD_VALUE, annotation.overFieldValue(), obj, true);
    }
    if (annotation.byFieldName().empty() == false) {
        m_Writer.addStringFieldReferenceToObj(BY_FIELD_NAME, annotation.byFieldName(), obj);
        m_Writer.addStringFieldReferenceToObj(BY_FIELD_VALUE,
                                              annotation.byFieldValue(), obj, true);
    }
}
}
}
