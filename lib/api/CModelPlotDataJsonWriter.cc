/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <api/CModelPlotDataJsonWriter.h>
#include <core/CLogger.h>
#include <core/CTimeUtils.h>

namespace ml {
namespace api {

// JSON field names
const std::string CModelPlotDataJsonWriter::JOB_ID("job_id");
const std::string CModelPlotDataJsonWriter::MODEL_PLOT("model_plot");
const std::string CModelPlotDataJsonWriter::DETECTOR_INDEX("detector_index");
const std::string CModelPlotDataJsonWriter::PARTITION_FIELD_NAME("partition_field_name");
const std::string CModelPlotDataJsonWriter::PARTITION_FIELD_VALUE("partition_field_value");
const std::string CModelPlotDataJsonWriter::TIME("timestamp");
const std::string CModelPlotDataJsonWriter::FEATURE("model_feature");
const std::string CModelPlotDataJsonWriter::BY("by");
const std::string CModelPlotDataJsonWriter::BY_FIELD_NAME("by_field_name");
const std::string CModelPlotDataJsonWriter::BY_FIELD_VALUE("by_field_value");
const std::string CModelPlotDataJsonWriter::OVER_FIELD_NAME("over_field_name");
const std::string CModelPlotDataJsonWriter::OVER_FIELD_VALUE("over_field_value");
const std::string CModelPlotDataJsonWriter::LOWER("model_lower");
const std::string CModelPlotDataJsonWriter::UPPER("model_upper");
const std::string CModelPlotDataJsonWriter::MEDIAN("model_median");
const std::string CModelPlotDataJsonWriter::ACTUAL("actual");
const std::string CModelPlotDataJsonWriter::BUCKET_SPAN("bucket_span");

CModelPlotDataJsonWriter::CModelPlotDataJsonWriter(core::CJsonOutputStreamWrapper &outStream)
    : m_Writer(outStream) {
}

void CModelPlotDataJsonWriter::writeFlat(const std::string &jobId, const model::CModelPlotData &data) {
    const std::string &partitionFieldName = data.partitionFieldName();
    const std::string &partitionFieldValue = data.partitionFieldValue();
    const std::string &overFieldName = data.overFieldName();
    const std::string &byFieldName = data.byFieldName();
    core_t::TTime     time = data.time();
    int               detectorIndex = data.detectorIndex();

    for (TFeatureStrByFieldDataUMapUMapCItr featureItr = data.begin();
         featureItr != data.end();
         ++featureItr) {
        std::string               feature = model_t::print(featureItr->first);
        const TStrByFieldDataUMap &byDataMap = featureItr->second;
        for (TStrByFieldDataUMapCItr byItr = byDataMap.begin(); byItr != byDataMap.end(); ++byItr) {
            const std::string     &    byFieldValue = byItr->first;
            const TByFieldData    &   byData = byItr->second;
            const TStrDoublePrVec &values = byData.s_ValuesPerOverField;
            if (values.empty()) {
                rapidjson::Value doc = m_Writer.makeObject();
                this->writeFlatRow(time, jobId, detectorIndex, partitionFieldName, partitionFieldValue, feature,
                                   byFieldName, byFieldValue, byData, data.bucketSpan(), doc);

                rapidjson::Value wrapper = m_Writer.makeObject();
                m_Writer.addMember(MODEL_PLOT, doc, wrapper);
                m_Writer.write(wrapper);
            } else {
                for (std::size_t valueIndex = 0; valueIndex < values.size(); ++valueIndex) {
                    const TStrDoublePr &keyValue = values[valueIndex];
                    rapidjson::Value   doc = m_Writer.makeObject();
                    this->writeFlatRow(time, jobId, detectorIndex, partitionFieldName, partitionFieldValue, feature,
                                       byFieldName, byFieldValue, byData, data.bucketSpan(), doc);
                    if (!overFieldName.empty()) {
                        m_Writer.addStringFieldCopyToObj(OVER_FIELD_NAME, overFieldName, doc);
                        m_Writer.addStringFieldCopyToObj(OVER_FIELD_VALUE, keyValue.first, doc, true);
                    }
                    m_Writer.addDoubleFieldToObj(ACTUAL, keyValue.second, doc);

                    rapidjson::Value wrapper = m_Writer.makeObject();
                    m_Writer.addMember(MODEL_PLOT, doc, wrapper);
                    m_Writer.write(wrapper);
                }
            }
        }
    }

    m_Writer.Flush();
}

void CModelPlotDataJsonWriter::writeFlatRow(core_t::TTime time,
                                            const std::string &jobId,
                                            int detectorIndex,
                                            const std::string &partitionFieldName,
                                            const std::string &partitionFieldValue,
                                            const std::string &feature,
                                            const std::string &byFieldName,
                                            const std::string &byFieldValue,
                                            const TByFieldData &byData,
                                            core_t::TTime bucketSpan,
                                            rapidjson::Value &doc) {
    m_Writer.addStringFieldCopyToObj(JOB_ID, jobId, doc, true);
    m_Writer.addIntFieldToObj(DETECTOR_INDEX, detectorIndex, doc);
    m_Writer.addStringFieldCopyToObj(FEATURE, feature, doc, true);
    // time is in Java format - milliseconds since the epoch
    m_Writer.addIntFieldToObj(TIME, time * 1000, doc);
    m_Writer.addIntFieldToObj(BUCKET_SPAN, bucketSpan, doc);
    if (!partitionFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(PARTITION_FIELD_NAME, partitionFieldName, doc);
        m_Writer.addStringFieldCopyToObj(PARTITION_FIELD_VALUE, partitionFieldValue, doc, true);
    }
    if (!byFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(BY_FIELD_NAME, byFieldName, doc);
        m_Writer.addStringFieldCopyToObj(BY_FIELD_VALUE, byFieldValue, doc, true);
    }
    m_Writer.addDoubleFieldToObj(LOWER, byData.s_LowerBound, doc);
    m_Writer.addDoubleFieldToObj(UPPER, byData.s_UpperBound, doc);
    m_Writer.addDoubleFieldToObj(MEDIAN, byData.s_Median, doc);
}

}
}
