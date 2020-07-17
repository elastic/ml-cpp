/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CSignal_h
#define INCLUDED_ml_maths_CSignal_h

#include <core/CContainerPrinter.h>
#include <core/CSmallVector.h>
#include <core/CVectorRange.h>

#include <maths/CBasicStatistics.h>
#include <maths/COrderings.h>
#include <maths/ImportExport.h>

#include <boost/optional.hpp>

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

namespace ml {
namespace maths {

//! \brief Useful functions from signal processing.
class MATHS_EXPORT CSignal {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleVec = std::vector<double>;
    using TComplex = std::complex<double>;
    using TComplexVec = std::vector<TComplex>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeSizePr2Vec = std::vector<TSizeSizePr>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TFloatMeanAccumulatorVecVec = std::vector<TFloatMeanAccumulatorVec>;
    using TFloatMeanAccumulatorCRng = core::CVectorRange<const TFloatMeanAccumulatorVec>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMeanAccumulatorVec1Vec = core::CSmallVector<TMeanAccumulatorVec, 1>;
    using TWeightFunction = std::function<double(std::size_t)>;
    using TPredictor = std::function<double(std::size_t)>;

    //! \brief A description of a seasonal component.
    struct SSeasonalComponentSummary {
        SSeasonalComponentSummary(std::size_t period,
                                  std::size_t startOfWeek,
                                  std::size_t windowRepeat,
                                  const TSizeSizePr& window)
            : s_Period{period}, s_StartOfWeek{startOfWeek},
              s_WindowRepeat{windowRepeat}, s_Window{window} {}

        bool operator==(const SSeasonalComponentSummary& rhs) const {
            return s_Period == rhs.s_Period && s_StartOfWeek == rhs.s_StartOfWeek &&
                   s_WindowRepeat == rhs.s_WindowRepeat && s_Window == rhs.s_Window;
        }

        bool operator<(const SSeasonalComponentSummary& rhs) const {
            return COrderings::lexicographical_compare(
                s_Period, s_StartOfWeek, s_WindowRepeat, s_Window, rhs.s_Period,
                rhs.s_StartOfWeek, rhs.s_WindowRepeat, rhs.s_Window);
        }

        bool contains(std::size_t index) const {
            index = (index - s_StartOfWeek) % s_WindowRepeat;
            return index >= s_Window.first && index < s_Window.second;
        }

        std::size_t offset(std::size_t index) const {
            index = (index - s_StartOfWeek) % s_WindowRepeat;
            return (index - s_Window.first) % this->period();
        }

        std::size_t period() const {
            return std::min(s_Period, s_Window.second - s_Window.first);
        }

        bool windowed() const {
            return s_Window.second - s_Window.first < s_WindowRepeat;
        }

        std::string print() const {
            return std::to_string(s_Period) + "/" + std::to_string(s_StartOfWeek) +
                   "/" + std::to_string(s_WindowRepeat) + "/" +
                   core::CContainerPrinter::print(s_Window);
        }

        std::size_t s_Period = 0;
        std::size_t s_StartOfWeek = 0;
        std::size_t s_WindowRepeat = 0;
        TSizeSizePr s_Window;
    };

    using TSeasonalComponentVec = std::vector<SSeasonalComponentSummary>;

public:
    //! Compute the conjugate of \p f.
    static void conj(TComplexVec& f);

    //! Compute the Hadamard product of \p fx and \p fy.
    static void hadamard(const TComplexVec& fx, TComplexVec& fy);

    //! Cooley-Tukey fast DFT transform implementation.
    //!
    //! \note This is a simple implementation radix 2 DIT which uses the chirp-z
    //! idea to handle the case that the length of \p fx is not a power of 2. As
    //! such it is definitely not a highly optimized FFT implementation. It should
    //! be sufficiently fast for our needs.
    static void fft(TComplexVec& f);

    //! This uses conjugate of the conjugate of the series is the inverse DFT trick
    //! to compute this using fft.
    static void ifft(TComplexVec& f);

    //! Compute the discrete cyclic autocorrelation of \p values for the offset
    //! \p offset.
    //!
    //! This is just
    //! <pre class="fragment">
    //!   \f$\(\frac{1}{(n-k)\sigma^2}\sum_{k=1}^{n-k}{(f(k) - \mu)(f(k+p) - \mu)}\)\f$
    //! </pre>
    //!
    //! \param[in] offset The offset as a distance in \p values.
    //! \param[in] values The values for which to compute the autocorrelation.
    static double cyclicAutocorrelation(std::size_t offset,
                                        const TFloatMeanAccumulatorVec& values);

    //! Compute the discrete cyclic autocorrelation of \p values for the offset
    //! \p offset.
    //!
    //! \note Implementation for vector ranges.
    static double cyclicAutocorrelation(std::size_t offset, TFloatMeanAccumulatorCRng values);

    //! Get linear autocorrelations for all offsets up to the length of \p values.
    //!
    //! \param[in] values The values for which to compute autocorrelation.
    //! \param[in] result Filled in with the autocorrelations of \p values for
    //! offsets 1, 2, ..., length \p values - 1.
    static void autocorrelations(const TFloatMeanAccumulatorVec& values, TDoubleVec& result);

    //! Compute the \p percentage percentile autocorrelation for a \p n normally
    //! distributed values.
    //!
    //! \param[in] percentage The required percentile in the range (0, 100).
    //! \param[in] autocorrelation The autocorrelation statistic.
    //! \param[in] n The number of values used to compute \p autocorrelation.
    static double
    autocorrelationAtPercentile(double percentage, double autocorrelation, double n);

    //! Remove the minimum MSE linear trend from \p values.
    //!
    //! \param[in] values The values from which to remove the linear trend.
    static void removeLinearTrend(TFloatMeanAccumulatorVec& values);

    //! Get the seasonal component summary with period \p period.
    static SSeasonalComponentSummary seasonalComponentSummary(std::size_t period);

    //! Append the summary of a seasonal component with period \p period to
    //! \p periods.
    //!
    //! \param[in] period The period to append.
    //! \param[out] periods The collection to update.
    static void appendSeasonalComponentSummary(std::size_t period,
                                               TSeasonalComponentVec& periods);

    //! Decompose the time series \p values into statistically significant seasonal
    //! components.
    //!
    //! This uses an F-test for the variance each component explains to see if there
    //! is strong statistical evidence for it.
    //!
    //! \param[in] values The time series to decompose.
    //! \param[in] outlierFraction The fraction of values treated as outliers.
    //! \param[in] week One week as a multiple of interval between \p values.
    //! \param[in] weight Controls the chance of selecting a specified period: the
    //! higher the weight the more likely it will be select in preference to one
    //! which is close.
    //! \return A summary of the seasonal components found.
    static TSeasonalComponentVec seasonalDecomposition(const TFloatMeanAccumulatorVec& values,
                                                       double outlierFraction,
                                                       std::size_t week,
                                                       const TWeightFunction& weight);

    //! Decompose the time series \p values into a weekday and weekend.
    //!
    //! This considers daily and weekly repeating patterns for each window of the
    //! decomposition and returns the decomposition for which there is the strongest
    //! evidence.
    //!
    //! \param[in] values The time series to decompose.
    //! \param[in] outlierFraction The fraction of values treated as outliers.
    //! \param[in] week One week as a multiple of interval between \p values.
    //! \return A summary of the best decomposition.
    static TSeasonalComponentVec tradingDayDecomposition(TFloatMeanAccumulatorVec values,
                                                         double outlierFraction,
                                                         std::size_t week);

    //! Fit the minimum MSE seasonal components with periods \p periods to \p values.
    //!
    //! \param[in] periods The seasonal components to fit.
    //! \param[in] values The time series to decompose.
    //! \param[out] components Filled in with the minimum MSE components with periods
    //! \p periods fit to \p values.
    static void fitSeasonalComponents(const TSeasonalComponentVec& periods,
                                      const TFloatMeanAccumulatorVec& values,
                                      TMeanAccumulatorVec1Vec& components);

    //! Fit the minimum MSE seasonal components with periods \p periods to \p values
    //! reweighting outliers in \p .
    //!
    //! \param[in] periods The seasonal components to fit.
    //! \param[in] outlierFraction The fraction of values treated as outliers.
    //! \param[in,out] values The time series to decompose whose outliers w.r.t. the
    //! seasonal components are reweighted.
    //! \param[out] components Filled in with the minimum MSE components with periods
    //! \p periods fit to \p values.
    static void fitSeasonalComponentsRobust(const TSeasonalComponentVec& periods,
                                            double outlierFraction,
                                            TFloatMeanAccumulatorVec& values,
                                            TMeanAccumulatorVec1Vec& components);

    //! Reweight outliers in \p values w.r.t. \p predictor.
    //!
    //! \param[in] predictor The \p values predictor.
    //! \param[in] fraction The fraction of values treated as outliers.
    //! \param[in,out] values The values to reweight.
    static void reweightOutliers(const TPredictor& predictor,
                                 double fraction,
                                 TFloatMeanAccumulatorVec& values);

    //! Compute the mean number of repeats of \p period in \p values.
    //!
    //! \param[in] values The values for which to compute the mean number of repeated
    //! values.
    //! \param[in] period The period for which to compute repeated values.
    //! \return The mean number of repeated values in \p values at each position
    //! in \p period.
    static double meanNumberRepeatedValues(const TFloatMeanAccumulatorVec& values,
                                           const SSeasonalComponentSummary& period);

    //! Compute the variance of \p values and the variance of \p values minus
    //! the predictions of \p components.
    //!
    //! \param[in] values The values from which to compute variances.
    //! \param[in] components The components for which to compute the residual
    //! variance.
    //! \return (total variance, residual variance).
    static TDoubleDoublePr residualVariance(const TFloatMeanAccumulatorVec& values,
                                            const TSeasonalComponentVec& periods,
                                            const TMeanAccumulatorVec1Vec& components);

    //! Compute the \p percentage percentile variance assuming it's calculated
    //! from \p degreesFreedom normal random variables.
    //!
    //! \param[in] percentage The required percentile in the range (0, 100).
    //! \param[in] variance The variance statistic value.
    //! \param[in] degreesFreedom The number of values used to compute \p variance.
    static double varianceAtPercentile(double percentage, double variance, double degreesFreedom);

    //! Choose the number of buckets to use for \p component for which there is
    //! the strongest supporting evidence in \p values.
    static std::size_t selectComponentSize(TFloatMeanAccumulatorVec values,
                                           double outlierFraction,
                                           std::size_t period);

    //! Restrict to the windows defined by \p period.
    //!
    //! \param[in] period Defines the window to restrict to if any.
    //! \param[in,out] values The values to restrict.
    static void restrictTo(const SSeasonalComponentSummary& period,
                           TFloatMeanAccumulatorVec& values);

    //! Erase \p values outside \p windows wrapping window indices which are out
    //! of bounds.
    //!
    //! \param[in] windows The subset set of \p values to copy.
    //! \param[in,out] values The values to restrict.
    //! \warning It's assumed that the restriction is one-to-one, i.e. the windows
    //! can't reference the same element of \p values twice or more.
    static void restrictTo(const TSizeSizePr2Vec& windows, TFloatMeanAccumulatorVec& values);

    //! Count the number of \p values which have non-zero count.
    template<typename CONTAINER>
    static std::size_t countNotMissing(const CONTAINER& values) {
        return std::count_if(values.begin(), values.end(), [](const auto& value) {
            return CBasicStatistics::count(value) > 0.0;
        });
    }

private:
    template<typename VALUES, typename COMPONENT>
    static void doFitSeasonalComponents(const TSeasonalComponentVec& periods,
                                        const VALUES& values,
                                        core::CSmallVector<COMPONENT, 1>& components);
    template<typename PREDICTOR, typename VALUES, typename COMPONENT>
    static void fitSeasonalComponentsMinusPrediction(const PREDICTOR& predictor,
                                                     const VALUES& values,
                                                     COMPONENT& component);
};
}
}

#endif // INCLUDED_ml_maths_CSignal_h
