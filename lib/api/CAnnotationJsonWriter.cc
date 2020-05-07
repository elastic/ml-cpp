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
const std::string ANNOTATION_RESULT_TYPE("annotation");
const std::string TIMESTAMP("timestamp");
const std::string END_TIMESTAMP("end_timestamp");
const std::string ANNOTATION("annotation");
const std::string CREATE_TIME("create_time");
const std::string CREATE_USERNAME("create_username");
const std::string MODIFIED_TIME("modified_time");
const std::string MODIFIED_USERNAME("modified_username");
const std::string TYPE("type");
const std::string JOB_ID("job_id");
const std::string DETECTOR_INDEX("detector_index");
const std::string PARTITION_FIELD_NAME("partition_field_name");
const std::string PARTITION_FIELD_VALUE("partition_field_value");
const std::string OVER_FIELD_NAME("over_field_name");
const std::string OVER_FIELD_VALUE("over_field_value");
const std::string BY_FIELD_NAME("by_field_name");
const std::string BY_FIELD_VALUE("by_field_value");
}

CAnnotationJsonWriter::CAnnotationJsonWriter(core::CJsonOutputStreamWrapper& outStream)
    : m_Writer(outStream) {
}

void CAnnotationJsonWriter::writeResult(const std::string& jobId,
                                        const model::CAnnotation& annotation) {

    rapidjson::Value doc = m_Writer.makeObject();
    write(jobId, annotation, doc);

    rapidjson::Value wrapper = m_Writer.makeObject();
    m_Writer.addMember(ANNOTATION_RESULT_TYPE, doc, wrapper);
    m_Writer.write(wrapper);
    m_Writer.Flush();
}

void CAnnotationJsonWriter::write(const std::string& jobId,
                                  const model::CAnnotation& annotation,
                                  rapidjson::Value& doc) {

    m_Writer.addStringFieldCopyToObj(JOB_ID, jobId, doc, true);
    m_Writer.addStringFieldCopyToObj(ANNOTATION, annotation.annotation(), doc);
    // time is in Java format - milliseconds since the epoch
    m_Writer.addTimeFieldToObj(TIMESTAMP, annotation.time(), doc);
    m_Writer.addTimeFieldToObj(END_TIMESTAMP, annotation.time(), doc);

    core_t::TTime currentTime(core::CTimeUtils::now());
    m_Writer.addTimeFieldToObj(CREATE_TIME, currentTime, doc);
    m_Writer.addTimeFieldToObj(MODIFIED_TIME, currentTime, doc);
    m_Writer.addStringFieldCopyToObj(CREATE_USERNAME, "_xpack", doc);
    m_Writer.addStringFieldCopyToObj(MODIFIED_USERNAME, "_xpack", doc);
    m_Writer.addStringFieldCopyToObj(TYPE, "annotation", doc);

    m_Writer.addIntFieldToObj(DETECTOR_INDEX, annotation.detectorIndex(), doc);
    if (!annotation.partitionFieldName().empty()) {
        m_Writer.addStringFieldCopyToObj(PARTITION_FIELD_NAME,
                                         annotation.partitionFieldName(), doc);
        m_Writer.addStringFieldCopyToObj(
            PARTITION_FIELD_VALUE, annotation.partitionFieldValue(), doc, true);
    }
    if (!annotation.overFieldName().empty()) {
        m_Writer.addStringFieldCopyToObj(OVER_FIELD_NAME, annotation.overFieldName(), doc);
        m_Writer.addStringFieldCopyToObj(OVER_FIELD_VALUE,
                                         annotation.overFieldValue(), doc, true);
    }
    if (!annotation.byFieldName().empty()) {
        m_Writer.addStringFieldCopyToObj(BY_FIELD_NAME, annotation.byFieldName(), doc);
        m_Writer.addStringFieldCopyToObj(BY_FIELD_VALUE,
                                         annotation.byFieldValue(), doc, true);
    }
}

}
}
