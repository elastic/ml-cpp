/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CForecastModelPersist_h
#define INCLUDED_ml_model_CForecastModelPersist_h

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>

#include <maths/CModel.h>

#include <model/CModelParams.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/filesystem.hpp>

#include <fstream>
#include <memory>

namespace ml {
namespace model {

//! \brief Persist/Restore CModel sub-classes to/from text representations for
//!  the purpose of forecasting.
//!
//! DESCRIPTION:\n
//! Persists/Restores models to disk for the purpose of restoring and forecasting
//! on them.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only as complete as required for forecasting.
//!
//! Persist and Restore are only done to avoid heap memory usage using temporary disk space.
//! No need for backwards compatibility and version'ing as code will only be used
//! locally never leaving process/io boundaries.
class MODEL_EXPORT CForecastModelPersist final {
public:
    using TMathsModelPtr = std::unique_ptr<maths::CModel>;

public:
    class MODEL_EXPORT CPersist final {
    public:
        explicit CPersist(const std::string& temporaryPath);

        //! add a model to persist
        void addModel(const maths::CModel* model,
                      core_t::TTime firstDataTime,
                      core_t::TTime lastDataTime,
                      const model_t::EFeature feature,
                      const std::string& byFieldValue);

        //! close the output file stream
        std::string finalizePersistAndGetFile();

    private:
        //! the filename to which to persist
        boost::filesystem::path m_FileName;

        //! the actual file where the models are persisted
        std::ofstream m_OutStream;

        //! number of models persisted
        size_t m_ModelCount;
    };

    class MODEL_EXPORT CRestore final {
    public:
        CRestore(const SModelParams& modelParams,
                 double minimumSeasonalVarianceScale,
                 const std::string& fileName);

        //! restore a single model
        bool nextModel(TMathsModelPtr& model,
                       core_t::TTime& firstDataTime,
                       core_t::TTime& lastDataTime,
                       model_t::EFeature& feature,
                       std::string& byFieldValue);

    private:
        //! model parameters required in order to restore the model
        const SModelParams m_ModelParams;

        //! minimum seasonal variance scale specific to the model
        double m_MinimumSeasonalVarianceScale;

        //! the actual file where the models are persisted
        std::ifstream m_InStream;

        //! the model state restorer
        core::CJsonStateRestoreTraverser m_RestoreTraverser;
    }; // class CRestore
};     // class CForecastModelPersist
}
}

#endif // INCLUDED_ml_model_CForecastModelPersist_h
