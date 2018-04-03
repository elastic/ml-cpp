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

#ifndef INCLUDED_ml_model_FrequencyPredicates_h
#define INCLUDED_ml_model_FrequencyPredicates_h

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>

namespace ml {
namespace model {

//! \brief Wrapper around personFrequency to test whether
//! frequency is greater than a specified frequency.
class MODEL_EXPORT CPersonFrequencyGreaterThan {
    public:
        CPersonFrequencyGreaterThan(const CAnomalyDetectorModel &model, double threshold);

        //! Test whether the person's frequency, whose identifier
        //! is the first element of \p t, is greater than the
        //! threshold supplied to the constructor.
        template<typename T>
        bool operator()(const std::pair<std::size_t, T> &t) {
            return m_Model->personFrequency(t.first) > m_Threshold;
        }

        //! Test whether the person's frequency, whose identifier
        //! is encoded in the first element of \p t, is greater
        //! than the threshold supplied to the constructor.
        template<typename T>
        bool operator()(const std::pair<std::pair<std::size_t, std::size_t>, T> &t) {
            return m_Model->personFrequency(CDataGatherer::extractPersonId(t)) > m_Threshold;
        }

    private:
        //! The model containing the person frequencies.
        const CAnomalyDetectorModel *m_Model;
        //! The test threshold frequency.
        double m_Threshold;
};

//! \brief Wrapper around personFrequency to test whether
//! frequency is greater than a specified frequency.
class MODEL_EXPORT CAttributeFrequencyGreaterThan {
    public:
        CAttributeFrequencyGreaterThan(const CAnomalyDetectorModel &model, double threshold);

        //! Test whether the person's frequency, whose identifier
        //! is encoded in the first element of \p t, is greater
        //! than the threshold supplied to the constructor.
        template<typename T>
        bool operator()(const std::pair<std::pair<std::size_t, std::size_t>, T> &t) {
            return m_Model->attributeFrequency(CDataGatherer::extractAttributeId(t)) > m_Threshold;
        }

    private:
        //! The model containing the person frequencies.
        const CAnomalyDetectorModel *m_Model;
        //! The test threshold frequency.
        double m_Threshold;
};

}
}

#endif // INCLUDED_ml_model_FrequencyPredicates_h
