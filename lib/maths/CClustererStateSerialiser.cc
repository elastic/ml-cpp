/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CClustererStateSerialiser.h>

#include <core/CLogger.h>

#include <maths/CKMeansOnline1d.h>
#include <maths/CXMeansOnline1d.h>

namespace ml {
namespace maths {

bool CClustererStateSerialiser::
operator()(const SDistributionRestoreParams& params, TClusterer1dPtr& ptr, core::CStateRestoreTraverser& traverser) {
    return this->operator()(params, CClusterer1d::CDoNothing(), CClusterer1d::CDoNothing(), ptr, traverser);
}

bool CClustererStateSerialiser::operator()(const SDistributionRestoreParams& params,
                                           const CClusterer1d::TSplitFunc& splitFunc,
                                           const CClusterer1d::TMergeFunc& mergeFunc,
                                           TClusterer1dPtr& ptr,
                                           core::CStateRestoreTraverser& traverser) {
    std::size_t numResults(0);

    do {
        const std::string& name = traverser.name();
        if (name == CClustererTypes::X_MEANS_ONLINE_1D_TAG) {
            ptr.reset(new CXMeansOnline1d(params, splitFunc, mergeFunc, traverser));
            ++numResults;
        } else if (name == CClustererTypes::K_MEANS_ONLINE_1D_TAG) {
            ptr.reset(new CKMeansOnline1d(params, traverser));
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

void CClustererStateSerialiser::operator()(const CClusterer1d& clusterer, core::CStatePersistInserter& inserter) {
    inserter.insertLevel(clusterer.persistenceTag(), boost::bind(&CClusterer1d::acceptPersistInserter, &clusterer, _1));
}
}
}
