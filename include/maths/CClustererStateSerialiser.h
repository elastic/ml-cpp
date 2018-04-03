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

#ifndef INCLUDED_ml_maths_CClustererStateSerialiser_h
#define INCLUDED_ml_maths_CClustererStateSerialiser_h

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CClusterer.h>
#include <maths/CXMeansOnlineFactory.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/shared_ptr.hpp>

namespace ml {
namespace maths {
template<typename T, std::size_t N>
class CXMeansOnline;
struct SDistributionRestoreParams;

//! \brief Convert CClusterer sub-classes to/from text representations.
//!
//! DESCRIPTION:\n
//! Encapsulate the conversion of arbitrary CClusterer sub-classes to/from
//! textual state.  In particular, the field name associated with each
//! clusterer type is then in one file.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The serialisation format must be a hierarchical format that supports
//! name/value pairs where the value may be a nested set of name/value
//! pairs.  Text format is used instead of binary because the format
//! needs to be storable in Elasticsearch. This also makes it easier to
//! provide backwards/forwards compatibility in the future as the classes
//! evolve.
//!
//! The field names given to each prior distribution class are deliberately
//! terse and uninformative to avoid giving away details of our analytics
//! to potential competitors.
class MATHS_EXPORT CClustererStateSerialiser {
public:
    typedef boost::shared_ptr<CClusterer1d> TClusterer1dPtr;

public:
    //! Construct the appropriate CClusterer sub-class from its state
    //! document representation.
    //!
    //! \note Sets \p ptr to NULL on failure.
    bool operator()(const SDistributionRestoreParams& params, TClusterer1dPtr& ptr, core::CStateRestoreTraverser& traverser);

    //! Construct the appropriate CClusterer sub-class from its state
    //! document representation.
    //!
    //! \note Sets \p ptr to NULL on failure.
    bool operator()(const SDistributionRestoreParams& params,
                    const CClusterer1d::TSplitFunc& splitFunc,
                    const CClusterer1d::TMergeFunc& mergeFunc,
                    TClusterer1dPtr& ptr,
                    core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to the supplied inserter.
    void operator()(const CClusterer1d& clusterer, core::CStatePersistInserter& inserter);

    //! Construct the appropriate CClusterer sub-class from its state
    //! document representation.
    //!
    //! \note Sets \p ptr to NULL on failure.
    template<typename T, std::size_t N>
    bool operator()(const SDistributionRestoreParams& params,
                    boost::shared_ptr<CClusterer<CVectorNx1<T, N>>>& ptr,
                    core::CStateRestoreTraverser& traverser) {
        return this->operator()(params, CClustererTypes::CDoNothing(), CClustererTypes::CDoNothing(), ptr, traverser);
    }

    //! Construct the appropriate CClusterer sub-class from its state
    //! document representation.
    //!
    //! \note Sets \p ptr to NULL on failure.
    template<typename T, std::size_t N>
    bool operator()(const SDistributionRestoreParams& params,
                    const CClustererTypes::TSplitFunc& splitFunc,
                    const CClustererTypes::TMergeFunc& mergeFunc,
                    boost::shared_ptr<CClusterer<CVectorNx1<T, N>>>& ptr,
                    core::CStateRestoreTraverser& traverser) {
        std::size_t numResults(0);

        do {
            const std::string& name = traverser.name();
            if (name == CClustererTypes::X_MEANS_ONLINE_TAG) {
                ptr.reset(CXMeansOnlineFactory::restore<T, N>(params, splitFunc, mergeFunc, traverser));
                ++numResults;
            } else {
                LOG_ERROR("No clusterer corresponds to node name " << traverser.name());
            }
        } while (traverser.next());

        if (numResults != 1) {
            LOG_ERROR("Expected 1 (got " << numResults << ") clusterer tags");
            ptr.reset();
            return false;
        }

        return true;
    }

    //! Persist state by passing information to the supplied inserter.
    template<typename T, std::size_t N>
    void operator()(const CClusterer<CVectorNx1<T, N>>& clusterer, core::CStatePersistInserter& inserter) {
        inserter.insertLevel(clusterer.persistenceTag(), boost::bind(&CClusterer<CVectorNx1<T, N>>::acceptPersistInserter, &clusterer, _1));
    }
};
}
}

#endif // INCLUDED_ml_maths_CClustererStateSerialiser_h
