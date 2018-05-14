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
    using TMathsModelPtr = std::shared_ptr<maths::CModel>;

public:
    class MODEL_EXPORT CPersist final {
    public:
        explicit CPersist(const std::string& temporaryPath);

        //! add a model to the persistence
        void addModel(const maths::CModel* model,
                      const model_t::EFeature feature,
                      const std::string& byFieldValue);

        //! close the outputStream
        const std::string& finalizePersistAndGetFile();

    private:
        static void persistOneModel(core::CStatePersistInserter& inserter,
                                    const maths::CModel* model,
                                    const model_t::EFeature feature,
                                    const std::string& byFieldValue);

    private:
        //! the filename where to persist to
        boost::filesystem::path m_FileName;

        //! the actual file where it models are persisted to
        std::ofstream m_OutStream;

        //! number of models persisted
        size_t m_ModelCount;
    };

    class MODEL_EXPORT CRestore final {
    public:
        explicit CRestore(const SModelParams& modelParams,
                          double minimumSeasonalVarianceScale,
                          const std::string& fileName);

        //! add a model to the persistence
        bool nextModel(TMathsModelPtr& model, model_t::EFeature& feature, std::string& byFieldValue);

    private:
        static bool restoreOneModel(core::CStateRestoreTraverser& traverser,
                                    SModelParams modelParams,
                                    double inimumSeasonalVarianceScale,
                                    TMathsModelPtr& model,
                                    model_t::EFeature& feature,
                                    std::string& byFieldValue);

    private:
        //! model parameters required in order to restore the model
        SModelParams m_ModelParams;

        //! minimum seasonal variance scale specific to the model
        double m_MinimumSeasonalVarianceScale;

        //! the actual file where it models are persisted to
        std::ifstream m_InStream;

        //! the persist inserter
        core::CJsonStateRestoreTraverser m_RestoreTraverser;
    }; // class CRestore
};     // class CForecastModelPersist
}
}

#endif // INCLUDED_ml_model_CForecastModelPersist_h
