/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CAnnotationJsonWriter.h>
#include <core/CLogger.h>
#include <core/CTimeUtils.h>

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
}

CAnnotationJsonWriter::CAnnotationJsonWriter(core::CJsonOutputStreamWrapper& outStream)
    : m_Writer(outStream) {
}

void CAnnotationJsonWriter::writeResult(const std::string& jobId,
                                        const model::CAnnotation& annotation) {

    rapidjson::Value obj = m_Writer.makeObject();
    populateAnnotationObject(jobId, annotation, obj);

    rapidjson::Value wrapper = m_Writer.makeObject();
    m_Writer.addMember(ANNOTATION_RESULT_TYPE, obj, wrapper);
    m_Writer.write(wrapper);
    m_Writer.Flush();
}

void CAnnotationJsonWriter::populateAnnotationObject(const std::string& jobId,
                                                     const model::CAnnotation& annotation,
                                                     rapidjson::Value& obj) {

    m_Writer.addStringFieldCopyToObj(JOB_ID, jobId, obj, true);
    m_Writer.addStringFieldCopyToObj(ANNOTATION, annotation.annotation(), obj);
    // time is in Java format - milliseconds since the epoch
    m_Writer.addTimeFieldToObj(TIMESTAMP, annotation.time(), obj);
    m_Writer.addTimeFieldToObj(END_TIMESTAMP, annotation.time(), obj);

    core_t::TTime currentTime(core::CTimeUtils::now());
    m_Writer.addTimeFieldToObj(CREATE_TIME, currentTime, obj);
    m_Writer.addTimeFieldToObj(MODIFIED_TIME, currentTime, obj);
    m_Writer.addStringFieldCopyToObj(CREATE_USERNAME, "_xpack", obj);
    m_Writer.addStringFieldCopyToObj(MODIFIED_USERNAME, "_xpack", obj);
    m_Writer.addStringFieldCopyToObj(TYPE, "annotation", obj);
    m_Writer.addStringFieldCopyToObj(EVENT, annotation.event(), obj);

    m_Writer.addIntFieldToObj(DETECTOR_INDEX, annotation.detectorIndex(), obj);
    if (annotation.partitionFieldName().empty() == false) {
        m_Writer.addStringFieldCopyToObj(PARTITION_FIELD_NAME,
                                         annotation.partitionFieldName(), obj);
        m_Writer.addStringFieldCopyToObj(
            PARTITION_FIELD_VALUE, annotation.partitionFieldValue(), obj, true);
    }
    if (annotation.overFieldName().empty() == false) {
        m_Writer.addStringFieldCopyToObj(OVER_FIELD_NAME, annotation.overFieldName(), obj);
        m_Writer.addStringFieldCopyToObj(OVER_FIELD_VALUE,
                                         annotation.overFieldValue(), obj, true);
    }
    if (annotation.byFieldName().empty() == false) {
        m_Writer.addStringFieldCopyToObj(BY_FIELD_NAME, annotation.byFieldName(), obj);
        m_Writer.addStringFieldCopyToObj(BY_FIELD_VALUE,
                                         annotation.byFieldValue(), obj, true);
    }
}

}
}
