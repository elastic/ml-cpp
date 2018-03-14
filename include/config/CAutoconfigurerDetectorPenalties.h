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

#ifndef INCLUDED_ml_config_CAutoconfigurerDetectorPenalties_h
#define INCLUDED_ml_config_CAutoconfigurerDetectorPenalties_h

#include <config/ImportExport.h>

#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>

namespace ml {
namespace config {
class CAutoconfigurerParams;
class CAutoconfigurerFieldRolePenalties;
class CDetectorSpecification;
class CPenalty;

//! \brief Defines the functions for penalizing detectors.
//!
//! DESCRIPTION:\n
//! This defines the penalties for a full detector specification based on the
//! data characteristics.
//!
//! IMPLEMENTATION:\n
//! This provides a single definition point for a logical group of penalties
//! and has been factored into its own class to avoid CAutoconfigurer becoming
//! monolithic.
class CONFIG_EXPORT CAutoconfigurerDetectorPenalties {
    public:
        typedef boost::shared_ptr<CPenalty> TPenaltyPtr;

    public:
        CAutoconfigurerDetectorPenalties(const CAutoconfigurerParams &params,
                                         const CAutoconfigurerFieldRolePenalties &fieldRolePenalties);

        //! Get the penalty for the detector \p spec.
        TPenaltyPtr penaltyFor(const CDetectorSpecification &spec);

    private:
        typedef boost::reference_wrapper<const CAutoconfigurerParams>             TAutoconfigurerParamsCRef;
        typedef boost::reference_wrapper<const CAutoconfigurerFieldRolePenalties> TAutoconfigurerFieldRolePenaltiesCRef;
        typedef std::vector<TPenaltyPtr>                                          TPenaltyPtrVec;

    private:
        //! Get the penalty for the detector \p spec based on its field roles.
        const CPenalty                        &fieldRolePenalty(const CDetectorSpecification &spec);

    private:
        //! The parameters.
        TAutoconfigurerParamsCRef             m_Params;

        //! The field role penalties.
        TAutoconfigurerFieldRolePenaltiesCRef m_FieldRolePenalties;

        //! The detector penalties based on their fields and roles.
        TPenaltyPtrVec                        m_DetectorFieldRolePenalties;

        //! The bucket length penalties.
        TPenaltyPtrVec                        m_BucketLengthPenalties;

        //! The function specific penalties.
        TPenaltyPtrVec                        m_FunctionSpecificPenalties;
};

}
}

#endif // INCLUDED_ml_config_CAutoconfigurerDetectorPenalties_h
