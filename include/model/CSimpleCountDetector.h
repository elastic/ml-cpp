/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CSimpleCountDetector_h
#define INCLUDED_ml_model_CSimpleCountDetector_h

#include <core/CoreTypes.h>

#include <model/CAnomalyDetector.h>
#include <model/ImportExport.h>

#include <string>

namespace ml {
namespace model {
class CAnomalyDetectorModelConfig;
class CLimits;

//! \brief
//! Simply records and reports overall event counts
//!
//! DESCRIPTION:\n
//! Given a stream of events, record and report the number observed
//! in each time bucket.
//!
//! The output of this class is used to plot event rate charts
//! and removes the burden of maintaining accurate overall event
//! counts from the real anomaly detector  classes.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Events must be received in increasing time order.
//!
//! This class sits within the CAnomalyDetector hierarchy even
//! though it doesn't detect anomalies, because its output goes
//! passes through the same data path as the output of the real
//! anomaly detector classes.
//!
class MODEL_EXPORT CSimpleCountDetector : public CAnomalyDetector {
public:
    CSimpleCountDetector(model_t::ESummaryMode summaryMode,
                         const CAnomalyDetectorModelConfig& modelConfig,
                         CLimits& limits,
                         const std::string& partitionFieldValue,
                         core_t::TTime firstTime,
                         const TModelFactoryCPtr& modelFactory);

    CSimpleCountDetector(bool isForPersistence, const CAnomalyDetector& other);

    //! Returns true.
    virtual bool isSimpleCount() const;

    //! Don't prune the simple count detector!
    virtual void pruneModels();

private:
    //! This function is called before adding a record allowing
    //! for varied preprocessing.
    virtual const TStrCPtrVec& preprocessFieldValues(const TStrCPtrVec& fieldValues);

private:
    //! Field values are strange compared to other anomaly detectors,
    //! because the "count" field always has value "count".  We need
    //! a vector to override the real value of any "count" field that
    //! might be present in the data.
    TStrCPtrVec m_FieldValues;
};
}
}

#endif // INCLUDED_ml_model_CSimpleCountDetector_h
