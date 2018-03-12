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

#ifndef INCLUDED_ml_config_CDetectorFieldRolePenalty_h
#define INCLUDED_ml_config_CDetectorFieldRolePenalty_h

#include <config/Constants.h>
#include <config/CPenalty.h>

#include <cstddef>

namespace ml {
namespace config {
class CAutoconfigurerParams;

//! \brief A penalty for a detector based on its field roles.
//!
//! DESCRIPTION:\n
//! This wraps up a collection of field role penalties and assigns
//! a penalty to a detector based on the product of all its argument
//! and partitioning fields penalties.
class CDetectorFieldRolePenalty : public CPenalty {
    public:
        CDetectorFieldRolePenalty(const CAutoconfigurerParams &params);

        //! Create a copy on the heap.
        virtual CDetectorFieldRolePenalty *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

        //! Set the field penalty for the field identified by \p index
        //! which must be one of ARGUMENT_INDEX, BY_INDEX, OVER_INDEX
        //! or PARTITION_INDEX.
        void addPenalty(std::size_t index, const CPenalty &penalty);

    private:
        //! Compute the penalty based on the detector's fields.
        virtual void penaltyFromMe(CDetectorSpecification &spec) const;

    private:
        //! The penalties to apply for each field.
        const CPenalty *m_FieldRolePenalties[constants::NUMBER_FIELD_INDICES];
};

}
}

#endif // INCLUDED_ml_config_CDetectorFieldRolePenalty_h
