/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_ml_maths_CTimeSeriesChangeDetector_h
#define INCLUDED_ml_maths_CTimeSeriesChangeDetector_h

#include <core/Constants.h>
#include <core/CoreTypes.h>
#include <core/CMemory.h>
#include <core/CSmallVector.h>
#include <core/CTriple.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{
class CModelAddSamplesParams;
class CPrior;
class CTimeSeriesDecompositionInterface;
struct SDistributionRestoreParams;
struct SModelRestoreParams;
struct STimeSeriesDecompositionRestoreParams;

namespace time_series_change_detector_detail
{
class CUnivariateChangeModel;
}

//! \brief A description of a time series change.
struct MATHS_EXPORT SChangeDescription
{
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDecompositionPtr = boost::shared_ptr<CTimeSeriesDecompositionInterface>;
    using TPriorPtr = boost::shared_ptr<CPrior>;

    //! The types of change we can detect.
    enum EDescription
    {
        E_LevelShift,
        E_TimeShift
    };

    SChangeDescription(EDescription decription,
                       double value,
                       const TPriorPtr &residualModel);

    //! Get a description of this change.
    std::string print() const;

    //! The type of change.
    EDescription s_Description;

    //! The change value.
    TDouble2Vec s_Value;

    //! Optionally, the trend model to use after the change.
    TDecompositionPtr s_TrendModel;

    //! The residual model to use after the change.
    TPriorPtr s_ResidualModel;
};

//! \brief Tests a variety of possible changes which might have
//! occurred in a time series and selects one if it provides a
//! good explanation of the recent behaviour.
class MATHS_EXPORT CUnivariateTimeSeriesChangeDetector
{
    public:
        using TDouble4Vec = core::CSmallVector<double, 4>;
        using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
        using TTimeDoublePr = std::pair<core_t::TTime, double>;
        using TTimeDoublePr1Vec = core::CSmallVector<TTimeDoublePr, 1>;
        using TTimeDoublePrCBuf = boost::circular_buffer<TTimeDoublePr>;
        using TWeightStyleVec = maths_t::TWeightStyleVec;
        using TDecompositionPtr = boost::shared_ptr<CTimeSeriesDecompositionInterface>;
        using TPriorPtr = boost::shared_ptr<CPrior>;
        using TOptionalChangeDescription = boost::optional<SChangeDescription>;

    public:
        CUnivariateTimeSeriesChangeDetector(const TDecompositionPtr &trendModel,
                                            const TPriorPtr &residualModel,
                                            core_t::TTime minimumTimeToDetect = 6 * core::constants::HOUR,
                                            core_t::TTime maximumTimeToDetect = core::constants::DAY,
                                            double minimumDeltaBicToDetect = 12.0);

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(const SModelRestoreParams &params,
                                    core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Check if there has been a change and get a description
        //! if there has been.
        TOptionalChangeDescription change();

        //! Add \p samples to the change detector.
        void addSamples(const TWeightStyleVec &weightStyles,
                        const TTimeDoublePr1Vec &samples,
                        const TDouble4Vec1Vec &weights);

        //! Check if we should stop testing.
        bool stopTesting() const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

    private:
        using TChangeModel = time_series_change_detector_detail::CUnivariateChangeModel;
        using TChangeModelPtr = boost::shared_ptr<TChangeModel>;
        using TChangeModelPtr4Vec = core::CSmallVector<TChangeModelPtr, 4>;
        using TMinMaxAccumulator = CBasicStatistics::CMinMax<core_t::TTime>;

    private:
        //! The minimum amount of time we need to observe before
        //! selecting a change model.
        core_t::TTime m_MinimumTimeToDetect;

        //! The maximum amount of time to try to detect a change.
        core_t::TTime m_MaximumTimeToDetect;

        //! The minimum increase in BIC select a change model.
        double m_MinimumDeltaBicToDetect;

        //! The start and end of the change model.
        TMinMaxAccumulator m_TimeRange;

        //! The count of samples added to the change models.
        std::size_t m_SampleCount;

        //! The current evidence of a change.
        double m_CurrentEvidenceOfChange;

        //! The change models.
        TChangeModelPtr4Vec m_ChangeModels;
};

namespace time_series_change_detector_detail
{

//! \brief Helper interface for change detection. Implementations of
//! this are used to model specific types of changes which can occur.
class MATHS_EXPORT CUnivariateChangeModel : private core::CNonCopyable
{
    public:
        using TDouble4Vec = core::CSmallVector<double, 4>;
        using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
        using TTimeDoublePr = std::pair<core_t::TTime, double>;
        using TTimeDoublePr1Vec = core::CSmallVector<TTimeDoublePr, 1>;
        using TTimeDoublePrCBuf = boost::circular_buffer<TTimeDoublePr>;
        using TWeightStyleVec = maths_t::TWeightStyleVec;
        using TDecompositionPtr = boost::shared_ptr<CTimeSeriesDecompositionInterface>;
        using TPriorPtr = boost::shared_ptr<CPrior>;
        using TOptionalChangeDescription = boost::optional<SChangeDescription>;

    public:
        CUnivariateChangeModel(const TDecompositionPtr &trendModel,
                               const TPriorPtr &residualModel);
        virtual ~CUnivariateChangeModel() = default;

        //! Initialize by reading state from \p traverser.
        virtual bool acceptRestoreTraverser(const SModelRestoreParams &params,
                                            core::CStateRestoreTraverser &traverser) = 0;

        //! Persist state by passing information to \p inserter.
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const = 0;

        //! The BIC of applying the change.
        virtual double bic() const = 0;

        //! Get a description of the change.
        virtual TOptionalChangeDescription change() const = 0;

        //! Update the change model with \p samples.
        virtual void addSamples(std::size_t count,
                                const TWeightStyleVec &weightStyles,
                                const TTimeDoublePr1Vec &samples,
                                const TDouble4Vec1Vec &weights) = 0;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

        //! Get the static size of this object.
        virtual std::size_t staticSize() const = 0;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed) const = 0;

    protected:
        //! The sample count to initialize a change model.
        static const std::size_t COUNT_TO_INITIALIZE = 5;

    protected:
        //! Restore the trend model reading state from \p traverser.
        bool restoreTrendModel(const STimeSeriesDecompositionRestoreParams &params,
                               core::CStateRestoreTraverser &traverser);

        //! Restore the residual model reading state from \p traverser.
        bool restoreResidualModel(const SDistributionRestoreParams &params,
                                  core::CStateRestoreTraverser &traverser);

        //! Get the log-likelihood.
        double logLikelihood() const;

        //! Update the data log-likelihood with \p logLikelihood.
        void addLogLikelihood(double logLikelihood);

        //! Get the time series trend model.
        const CTimeSeriesDecompositionInterface &trendModel() const;
        //! Get the time series trend model.
        CTimeSeriesDecompositionInterface &trendModel();

        //! Get the time series residual model.
        const CPrior &residualModel() const;
        //! Get the time series residual model.
        CPrior &residualModel();
        //! Get the time series residual model member variable.
        TPriorPtr residualModelPtr() const;

    private:
        //! The likelihood of the data under this model.
        double m_LogLikelihood;

        //! A model decomposing the time series trend.
        TDecompositionPtr m_TrendModel;

        //! A reference to the underlying prior.
        TPriorPtr m_ResidualModel;
};

//! \brief Used to capture the likelihood of the data given no change.
class MATHS_EXPORT CUnivariateNoChangeModel final : public CUnivariateChangeModel
{
    public:
        CUnivariateNoChangeModel(const TDecompositionPtr &trendModel,
                                 const TPriorPtr &residualModel);

        //! Initialize by reading state from \p traverser.
        virtual bool acceptRestoreTraverser(const SModelRestoreParams &params,
                                            core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Returns the no change BIC.
        virtual double bic() const;

        //! Returns a null object.
        virtual TOptionalChangeDescription change() const;

        //! Get the log likelihood of \p samples.
        virtual void addSamples(std::size_t count,
                                const TWeightStyleVec &weightStyles,
                                const TTimeDoublePr1Vec &samples,
                                const TDouble4Vec1Vec &weights);

        //! Get the static size of this object.
        virtual std::size_t staticSize() const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed) const;

    private:
};

//! \brief Captures the likelihood of the data given an arbitrary
//! level shift.
class MATHS_EXPORT CUnivariateLevelShiftModel final : public CUnivariateChangeModel
{
    public:
        CUnivariateLevelShiftModel(const TDecompositionPtr &trendModel,
                                   const TPriorPtr &residualModel);

        //! Initialize by reading state from \p traverser.
        virtual bool acceptRestoreTraverser(const SModelRestoreParams &params,
                                            core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! The BIC of applying the level shift.
        virtual double bic() const;

        //! Get a description of the level shift.
        virtual TOptionalChangeDescription change() const;

        //! Update with \p samples.
        virtual void addSamples(std::size_t count,
                                const TWeightStyleVec &weightStyles,
                                const TTimeDoublePr1Vec &samples,
                                const TDouble4Vec1Vec &weights);

        //! Get the static size of this object.
        virtual std::size_t staticSize() const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed) const;

    private:
        using TDoubleVec = std::vector<double>;
        using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    private:
        //! The optimal shift.
        TMeanAccumulator m_Shift;

        //! Get the number of samples.
        double m_SampleCount;
};

//! \brief Captures the likelihood of the data given a specified
//! time shift.
class MATHS_EXPORT CUnivariateTimeShiftModel final : public CUnivariateChangeModel
{
    public:
        CUnivariateTimeShiftModel(const TDecompositionPtr &trendModel,
                                  const TPriorPtr &residualModel,
                                  core_t::TTime shift);

        //! Initialize by reading state from \p traverser.
        virtual bool acceptRestoreTraverser(const SModelRestoreParams &params,
                                            core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! The BIC of applying the time shift.
        virtual double bic() const;

        //! Get a description of the time shift.
        virtual TOptionalChangeDescription change() const;

        //! Update with \p samples.
        virtual void addSamples(std::size_t count,
                                const TWeightStyleVec &weightStyles,
                                const TTimeDoublePr1Vec &samples,
                                const TDouble4Vec1Vec &weights);

        //! Get the static size of this object.
        virtual std::size_t staticSize() const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed) const;

    private:
        //! The shift in time of the time series trend model.
        core_t::TTime m_Shift;
};

}

}
}

#endif // INCLUDED_ml_maths_CTimeSeriesChangeDetector_h
