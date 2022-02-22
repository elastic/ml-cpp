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

#include <maths/common/CClustererStateSerialiser.h>

#include <core/CLogger.h>

#include <maths/common/CKMeansOnline1d.h>
#include <maths/common/CXMeansOnline1d.h>

namespace ml {
namespace maths {
namespace common {
bool CClustererStateSerialiser::operator()(const SDistributionRestoreParams& params,
                                           TClusterer1dPtr& ptr,
                                           core::CStateRestoreTraverser& traverser) const {
    return this->operator()(params, CClusterer1d::CDoNothing(),
                            CClusterer1d::CDoNothing(), ptr, traverser);
}

bool CClustererStateSerialiser::operator()(const SDistributionRestoreParams& params,
                                           const CClusterer1d::TSplitFunc& splitFunc,
                                           const CClusterer1d::TMergeFunc& mergeFunc,
                                           TClusterer1dPtr& ptr,
                                           core::CStateRestoreTraverser& traverser) const {
    std::size_t numResults(0);

    do {
        const std::string& name = traverser.name();
        if (name == CClustererTypes::X_MEANS_ONLINE_1D_TAG) {
            ptr = std::make_unique<CXMeansOnline1d>(params, splitFunc, mergeFunc, traverser);
            ++numResults;
        } else if (name == CClustererTypes::K_MEANS_ONLINE_1D_TAG) {
            ptr = std::make_unique<CKMeansOnline1d>(params, traverser);
            ++numResults;
        } else {
            LOG_ERROR(<< "No clusterer corresponds to node name " << traverser.name());
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR(<< "Expected 1 (got " << numResults << ") clusterer tags");
        ptr.reset();
        return false;
    }

    return true;
}

void CClustererStateSerialiser::operator()(const CClusterer1d& clusterer,
                                           core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(clusterer.persistenceTag(), [&](auto& inserter_) {
        clusterer.acceptPersistInserter(inserter_);
    });
}
}
}
}
