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

#ifndef INCLUDED_ml_maths_CMultivariateOneOfNPriorFactory_h
#define INCLUDED_ml_maths_CMultivariateOneOfNPriorFactory_h

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/shared_ptr.hpp>

#include <cstddef>

namespace ml
{
namespace core
{
class CStateRestoreTraverser;
}

namespace maths
{
class CMultivariatePrior;
struct SDistributionRestoreParams;

//! \brief Factory for multivariate 1-of-n priors.
class MATHS_EXPORT CMultivariateOneOfNPriorFactory
{
    public:
        typedef boost::shared_ptr<CMultivariatePrior> TPriorPtr;
        typedef std::vector<TPriorPtr> TPriorPtrVec;

    public:
        //! Create a new non-informative multivariate normal prior.
        static TPriorPtr nonInformative(std::size_t dimension,
                                        maths_t::EDataType dataType,
                                        double decayRate,
                                        const TPriorPtrVec &models);

        //! Create reading state from its state document representation.
        static bool restore(std::size_t dimension,
                            const SDistributionRestoreParams &params,
                            TPriorPtr &ptr,
                            core::CStateRestoreTraverser &traverser);
};

}
}

#endif // INCLUDED_ml_maths_CMultivariateOneOfNPriorFactory_h
