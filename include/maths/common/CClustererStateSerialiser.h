/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_maths_common_CClustererStateSerialiser_h
#define INCLUDED_ml_maths_common_CClustererStateSerialiser_h

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/common/CClusterer.h>
#include <maths/common/CXMeansOnlineFactory.h>
#include <maths/common/ImportExport.h>
#include <maths/common/MathsTypes.h>

#include <functional>
#include <memory>

namespace ml {
namespace maths {
namespace common {
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
class MATHS_COMMON_EXPORT CClustererStateSerialiser {
public:
    using TClusterer1dPtr = std::unique_ptr<CClusterer1d>;

public:
    //! Construct the appropriate CClusterer sub-class from its state
    //! document representation.
    //!
    //! \note Sets \p ptr to NULL on failure.
    bool operator()(const SDistributionRestoreParams& params,
                    TClusterer1dPtr& ptr,
                    core::CStateRestoreTraverser& traverser) const;

    //! Construct the appropriate CClusterer sub-class from its state
    //! document representation.
    //!
    //! \note Sets \p ptr to NULL on failure.
    bool operator()(const SDistributionRestoreParams& params,
                    const CClusterer1d::TSplitFunc& splitFunc,
                    const CClusterer1d::TMergeFunc& mergeFunc,
                    TClusterer1dPtr& ptr,
                    core::CStateRestoreTraverser& traverser) const;

    //! Persist state by passing information to the supplied inserter.
    void operator()(const CClusterer1d& clusterer, core::CStatePersistInserter& inserter) const;

    //! Construct the appropriate CClusterer sub-class from its state
    //! document representation.
    //!
    //! \note Sets \p ptr to NULL on failure.
    template<typename T, std::size_t N>
    bool operator()(const SDistributionRestoreParams& params,
                    std::unique_ptr<CClusterer<CVectorNx1<T, N>>>& ptr,
                    core::CStateRestoreTraverser& traverser) const {
        return this->operator()(params, CClustererTypes::CDoNothing(),
                                CClustererTypes::CDoNothing(), ptr, traverser);
    }

    //! Construct the appropriate CClusterer sub-class from its state
    //! document representation.
    //!
    //! \note Sets \p ptr to NULL on failure.
    template<typename T, std::size_t N>
    bool operator()(const SDistributionRestoreParams& params,
                    const CClustererTypes::TSplitFunc& splitFunc,
                    const CClustererTypes::TMergeFunc& mergeFunc,
                    std::unique_ptr<CClusterer<CVectorNx1<T, N>>>& ptr,
                    core::CStateRestoreTraverser& traverser) const {
        std::size_t numResults(0);

        do {
            const std::string& name = traverser.name();
            if (name == CClustererTypes::X_MEANS_ONLINE_TAG) {
                ptr.reset(CXMeansOnlineFactory::restore<T, N>(params, splitFunc,
                                                              mergeFunc, traverser));
                ++numResults;
            } else {
                LOG_ERROR(<< "No clusterer corresponds to node name "
                          << traverser.name());
            }
        } while (traverser.next());

        if (numResults != 1) {
            LOG_ERROR(<< "Expected 1 (got " << numResults << ") clusterer tags");
            ptr.reset();
            return false;
        }

        return true;
    }

    //! Persist state by passing information to the supplied inserter.
    template<typename T, std::size_t N>
    void operator()(const CClusterer<CVectorNx1<T, N>>& clusterer,
                    core::CStatePersistInserter& inserter) const {
        inserter.insertLevel(clusterer.persistenceTag(), [&clusterer](auto& inserter_) {
            clusterer.acceptPersistInserter(inserter_);
        });
    }
};
}
}
}

#endif // INCLUDED_ml_maths_common_CClustererStateSerialiser_h
