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

#ifndef INCLUDED_ml_config_CLowInformationContentPenalty_h
#define INCLUDED_ml_config_CLowInformationContentPenalty_h

#include <config/CPenalty.h>
#include <config/ImportExport.h>

namespace ml
{
namespace config
{

//! \brief A penalty for the information content command if there is
//! little evidence that the categories are carrying any information.
//!
//! DESCRIPTION:\n
//! The information content command should only be suggested when there
//! is significant evidence that the categories of a field could be being
//! used to transmit information. We check to see if there is:
//!   -# Much variation in the category lengths.
//!   -# Any long categories.
//!   -# Significant empirical entropy in the categories relative to
//!      their distinct count (which bounds the entropy).
class CONFIG_EXPORT CLowInformationContentPenalty : public CPenalty
{
    public:
        CLowInformationContentPenalty(const CAutoconfigurerParams &params);

        //! Create a copy on the heap.
        virtual CLowInformationContentPenalty *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

    private:
        //! Compute a penalty for rare detectors.
        virtual void penaltyFromMe(CDetectorSpecification &spec) const;
};

}
}

#endif // INCLUDED_ml_config_CLowInformationContentPenalty_h
