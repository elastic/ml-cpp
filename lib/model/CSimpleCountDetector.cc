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
#include <model/CSimpleCountDetector.h>

#include <maths/CIntegerTools.h>

#include <model/CAnomalyDetectorModelConfig.h>


namespace ml {
namespace model {

CSimpleCountDetector::CSimpleCountDetector(int detectorIndex,
                                           model_t::ESummaryMode summaryMode,
                                           const CAnomalyDetectorModelConfig &modelConfig,
                                           CLimits &limits,
                                           const std::string &partitionFieldValue,
                                           core_t::TTime firstTime,
                                           const TModelFactoryCPtr &modelFactory)
    : CAnomalyDetector(detectorIndex,
                       limits,
                       modelConfig,
                       partitionFieldValue,
                       firstTime,
                       modelFactory),
      m_FieldValues(summaryMode == model_t::E_None ? 1 : 2) {
    // We use a single event rate detector to maintain the counts, and for the
    // special case of the simple count detector, we'll create it before we've
    // seen any events, so that in the extreme case of no events in a search
    // period, there is still a model recording the zero count
    this->initSimpleCounting();
}

CSimpleCountDetector::CSimpleCountDetector(bool isForPersistence,
                                           const CAnomalyDetector &other)
    : CAnomalyDetector(isForPersistence, other) {
}

bool CSimpleCountDetector::isSimpleCount(void) const {
    return true;
}

void CSimpleCountDetector::pruneModels(void) {
    return;
}

const CAnomalyDetector::TStrCPtrVec &
CSimpleCountDetector::preprocessFieldValues(const TStrCPtrVec &fieldValues) {
    // The first field value is always the magic word "count", but for
    // summarised input we need to pass on the true value of the second field
    if (m_FieldValues.size() > 1) {
        m_FieldValues[1] = (fieldValues.size() > 1) ? fieldValues[1] : &EMPTY_STRING;
    }

    return m_FieldValues;
}


}
}

