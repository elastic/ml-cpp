/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CSignal_h
#define INCLUDED_ml_maths_CSignal_h

#include <core/CContainerPrinter.h>
#include <core/CVectorRange.h>

#include <maths/CBasicStatistics.h>
#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>
#include <maths/ImportExport.h>

#include <boost/optional.hpp>

#include <algorithm>
#include <complex>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace ml {
namespace maths {

//! \brief Useful functions from signal processing.
class MATHS_EXPORT CSignal {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using TSizeSizeSizeTr = std::tuple<std::size_t, std::size_t, std::size_t>;
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TComplex = std::complex<double>;
    using TComplexVec = std::vector<TComplex>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeSizePr2Vec = std::vector<TSizeSizePr>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TFloatMeanAccumulatorVecVec = std::vector<TFloatMeanAccumulatorVec>;
    using TFloatMeanAccumulatorCRng = core::CVectorRange<const TFloatMeanAccumulatorVec>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;
    using TMomentTransformFunc = std::function<double(const TFloatMeanAccumulator&)>;
    using TMomentWeightFunc = std::function<double(const TFloatMeanAccumulator&)>;
    using TIndexWeightFunc = std::function<double(std::size_t)>;
    using TPredictor = std::function<double(std::size_t)>;

    //! \brief A description of a seasonal component.
    struct SSeasonalComponentSummary {
        SSeasonalComponentSummary() = default;
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
                s_Period, s_StartOfWeek, s_WindowRepeat, s_Window, // lhs
                rhs.s_Period, rhs.s_StartOfWeek, rhs.s_WindowRepeat, rhs.s_Window);
        }

        //! Check equality to within a relative \p eps.
        bool almostEqual(const SSeasonalComponentSummary& rhs, double eps) const {
            return almostEqual(s_Period, rhs.s_Period, eps) &&
                   almostEqual(s_StartOfWeek, rhs.s_StartOfWeek, eps) &&
                   almostEqual(s_WindowRepeat, rhs.s_WindowRepeat, eps) &&
                   almostEqual(s_Window.first, rhs.s_Window.first, eps) &&
                   almostEqual(s_Window.second, rhs.s_Window.second, eps);
        }
        //! Check equality of the period to within a relative \p eps.
        bool periodAlmostEqual(const SSeasonalComponentSummary& rhs, double eps) const {
            return almostEqual(s_Period, rhs.s_Period, eps);
        }
        //! Check equality to within a relative \p eps.
        static bool almostEqual(std::size_t i, std::size_t j, double eps) {
            std::size_t max{std::max(i, j)};
            std::size_t min{std::min(i, j)};
            return static_cast<double>(max - min) <= eps * static_cast<double>(max);
        }

        //! If this is windowed check whether the window contains \p index.
        bool contains(std::size_t index) const {
            index = (s_WindowRepeat + index - s_StartOfWeek) % s_WindowRepeat;
            return index >= s_Window.first && index < s_Window.second;
        }

        //! Get the fraction of \p range which falls in the time windows.
        std::size_t fractionInWindow(std::size_t range) const {
            if (s_WindowRepeat > 0) {
                return ((s_Window.second - s_Window.first) * range) / s_WindowRepeat;
            }
            return range;
        }

        //! The offset of \p index in the period.
        std::size_t offset(std::size_t index) const {
            index = (s_WindowRepeat + index - s_StartOfWeek) % s_WindowRepeat;
            return (index - s_Window.first) % this->period();
        }

        //! The next repeat of the value at \p index.
        std::size_t nextRepeat(std::size_t index) const {
            for (index += s_Period; this->contains(index) == false; index += s_Period) {
            }
            return index;
        }

        //! The windowed length of the repeat.
        std::size_t period() const {
            return std::min(s_Period, s_Window.second - s_Window.first);
        }

        //! True if this component is restricted to time windows.
        bool windowed() const {
            return s_Window.second - s_Window.first < s_WindowRepeat;
        }

        //! The indices of the start and end of the values in the time windows.
        TSizeSizePr2Vec windows(std::size_t numberValues) const {
            TSizeSizePr2Vec result;
            if (this->windowed()) {
                result.reserve((numberValues + s_WindowRepeat - 1) / s_WindowRepeat);
                for (std::size_t i = 0; i < numberValues; i += s_WindowRepeat) {
                    result.emplace_back(s_StartOfWeek + s_Window.first + i,
                                        s_StartOfWeek + s_Window.second + i);
                }
            } else {
                result.emplace_back(0, numberValues);
            }
            return result;
        }

        //! The value of the \p model of this component at \p index.
        template<typename MODEL>
        double value(const MODEL& model, std::size_t index) const {
            return this->contains(index) == false
                       ? 0.0
                       : CBasicStatistics::mean(model[this->offset(index)]);
        }

        //! Get a description of the component.
        std::string print() const {
            std::ostringstream result;
            result << s_Period << "/" << s_StartOfWeek << "/" << s_WindowRepeat
                   << "/" << core::CContainerPrinter::print(s_Window);
            return result.str();
        }

        std::size_t s_Period = 0;
        std::size_t s_StartOfWeek = 0;
        std::size_t s_WindowRepeat = 0;
        TSizeSizePr s_Window;
    };

    using TSeasonalComponentVec = std::vector<SSeasonalComponentSummary>;

    //! \brief The variance statistics associated with collection of one or more
    //! seasonal components.
    struct SVarianceStats {
        //! Get a description of the variance stats.
        std::string print() const {
            std::ostringstream result;
            result << "v = " << s_ResidualVariance << ", <v> = " << s_TruncatedResidualVariance
                   << ", f = " << s_DegreesFreedom;
            return result.str();
        }

        double s_ResidualVariance;
        double s_TruncatedResidualVariance;
        double s_DegreesFreedom;
        std::size_t s_NumberParameters;
    };

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
    //! \param[in] period The seasonal component for which to compute autocorrelation.
    //! \param[in] values The values for which to compute the autocorrelation.
    //! \param[in] transform Transforms \p values before computing the autocorrelation.
    //! \param[in] weight Weights \p values for computing the autocorrelation.
    //! \param[in] eps The minimum variance we care about. This is added to the variance
    //! when computing the autocorrelation so if the data variance is less than this it
    //! drives the autocorrelation smoothly to zero.
    static double cyclicAutocorrelation(const SSeasonalComponentSummary& period,
                                        const TFloatMeanAccumulatorVec& values,
                                        const TMomentTransformFunc& transform = mean,
                                        const TMomentWeightFunc& weight = count,
                                        double eps = 0.0);

    //! Compute the discrete cyclic autocorrelation of \p values for the offset
    //! \p offset.
    //!
    //! \note Implementation for vector ranges.
    static double cyclicAutocorrelation(const SSeasonalComponentSummary& period,
                                        const TFloatMeanAccumulatorCRng& values,
                                        const TMomentTransformFunc& transform = mean,
                                        const TMomentWeightFunc& weight = count,
                                        double eps = 0.0);

    //! Get linear autocorrelations for all offsets up to the length of \p values.
    //!
    //! \param[in] values The values for which to compute autocorrelation.
    //! \param[in,out] f Placeholder for the function for which to compute
    //! autocorrelations to avoid repeatedly allocating the vector if this
    //! is called in a loop.
    //! \param[out] result Filled in with the autocorrelations of \p values for
    //! offsets 1, 2, ..., length \p values - 1.
    static void autocorrelations(const TFloatMeanAccumulatorVec& values,
                                 TComplexVec& f,
                                 TDoubleVec& result);

    //! Overload which creates a temporary function.
    static void autocorrelations(const TFloatMeanAccumulatorVec& values, TDoubleVec& result) {
        TComplexVec f;
        autocorrelations(values, f, result);
    }

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
    //! \param[in,out] values The time series to decompose. Outliers are reweighted
    //! if \p outlierFraction is greater than zero.
    //! \param[in] outlierFraction The fraction of values treated as outliers.
    //! \param[in] diurnal (day, week, year) as multiples of the time interval
    //! between \p values.
    //! \param[in] startOfWeekOverride The start of the week to use if one is known.
    //! \param[in] significantPValue The p-value at which to return a component.
    //! \param[in] maxComponents The maximum number of modelled components.
    //! \return A summary of the seasonal components found.
    static TSeasonalComponentVec
    seasonalDecomposition(const TFloatMeanAccumulatorVec& values,
                          double outlierFraction,
                          const TSizeSizeSizeTr& diurnal,
                          TOptionalSize startOfWeekOverride = TOptionalSize{},
                          double significantPValue = 0.05,
                          std::size_t maxComponents = 10);

    //! Decompose the time series \p values into a weekday and weekend.
    //!
    //! This considers daily and weekly repeating patterns for each window of the
    //! decomposition and returns the decomposition for which there is the strongest
    //! evidence.
    //!
    //! \param[in,out] values The time series to decompose. Outliers are reweighted
    //! if \p outlierFraction is greater than zero.
    //! \param[in] outlierFraction The fraction of values treated as outliers.
    //! \param[in] week One week as a multiple of interval between \p values.
    //! \param[in] startOfWeekOverride The start of the week to use if one is known.
    //! \param[in] significantPValue The p-value at which to return a component.
    //! \return A summary of the best decomposition.
    static TSeasonalComponentVec
    tradingDayDecomposition(const TFloatMeanAccumulatorVec& values,
                            double outlierFraction,
                            std::size_t week,
                            TOptionalSize startOfWeekOverride = TOptionalSize{},
                            double significantPValue = 0.05);

    //! Fit the minimum MSE seasonal components with periods \p periods to \p values.
    //!
    //! \param[in] periods The seasonal components to fit.
    //! \param[in] values The time series to decompose.
    //! \param[out] components Filled in with the minimum MSE components with periods
    //! \p periods fit to \p values.
    static void fitSeasonalComponents(const TSeasonalComponentVec& periods,
                                      const TFloatMeanAccumulatorVec& values,
                                      TMeanAccumulatorVecVec& components);

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
                                            TMeanAccumulatorVecVec& components);

    //! Reweight outliers in \p values w.r.t. \p components.
    //!
    //! \param[in] periods The seasonal components in \p values.
    //! \param[in] components The seasonal component models in \p values.
    //! \param[in] fraction The fraction of values treated as outliers.
    //! \param[in,out] values The values to reweight.
    //! \return True if this reweighted any \p values.
    static bool reweightOutliers(const TSeasonalComponentVec& periods,
                                 const TMeanAccumulatorVecVec& components,
                                 double fraction,
                                 TFloatMeanAccumulatorVec& values);

    //! Reweight outliers in \p values w.r.t. \p predictor.
    //!
    //! \param[in] predictor The \p values predictor.
    //! \param[in] fraction The fraction of values treated as outliers.
    //! \param[in,out] values The values to reweight.
    //! \return True if this reweighted any \p values.
    static bool reweightOutliers(const TPredictor& predictor,
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

    //! Compute statistics related to the variance of \p values minus the predictions
    //! of \p components
    //!
    //! \param[in] values The values from which to compute variances.
    //! \param[in] periods The seasonal components being modelled.
    //! \param[in] components The models of \p periods to use.
    //! \return The variance weighted by value counts and the degrees of
    //! freedom in the variance estimate.
    static SVarianceStats residualVarianceStats(const TFloatMeanAccumulatorVec& values,
                                                const TSeasonalComponentVec& periods,
                                                const TMeanAccumulatorVecVec& components);

    //! Compute the variance of \p values minus the predictions of \p components.
    //!
    //! \param[in] values The values from which to compute variances.
    //! \param[in] periods The seasonal components being modelled.
    //! \param[in] components The models of \p periods to use.
    //! \param[in] weight The function which weights values when computing variances.
    //! \return The residual variance.
    static double residualVariance(const TFloatMeanAccumulatorVec& values,
                                   const TSeasonalComponentVec& periods,
                                   const TMeanAccumulatorVecVec& components,
                                   const TMomentWeightFunc& weight = count);

    //! Get the p-value of the variance statistics \p H1 given the null \p H0.
    //!
    //! \param[in] H0 The variance statistics of the null hypothesis.
    //! \param[in] H1 The variance statistics of the alternative hypothesis.
    //! \note This assumes that \p H1 nests inside \p H0.
    //! \return The p-value of \p H1.
    static double nestedDecompositionPValue(const SVarianceStats& H0,
                                            const SVarianceStats& H1);

    //! Choose the number of buckets to use for a component model with \p period
    //! for which there is the strongest supporting evidence in \p values.
    //!
    //! \param[in] values The values to which to fit the component.
    //! \param[in] period The component's period.
    //! \return The number of buckets.
    static std::size_t selectComponentSize(const TFloatMeanAccumulatorVec& values,
                                           std::size_t period);

    //! Restrict to the windows defined by \p period.
    //!
    //! \param[in] period Defines the window to restrict to if any.
    //! \param[in,out] values The values to restrict.
    //! \tparam CONTAINER  A std compliant random access container.
    template<typename CONTAINER>
    static void restrictTo(const SSeasonalComponentSummary& period, CONTAINER& values) {
        if (period.windowed()) {
            values.resize(CIntegerTools::floor(values.size(), period.s_WindowRepeat));
            restrictTo(period.windows(values.size()), values);
        }
    }

    //! Erase \p values outside \p windows wrapping window indices which are out
    //! of bounds.
    //!
    //! \param[in] windows The subset set of \p values to copy.
    //! \param[in,out] values The values to restrict.
    //! \warning It's assumed that the restriction is one-to-one, i.e. the windows
    //! can't reference the same element of \p values twice or more.
    //! \tparam CONTAINER  A std compliant random access container.
    template<typename CONTAINER>
    static void restrictTo(const TSizeSizePr2Vec& windows, CONTAINER& values) {
        if (values.size() == 0) {
            return;
        }
        if (windows.empty()) {
            values.clear();
            return;
        }

        std::size_t m{std::accumulate(windows.begin(), windows.end(), std::size_t{0},
                                      [](std::size_t m_, const auto& window) {
                                          return m_ + window.second - window.first;
                                      })};
        std::size_t n{values.size()};

        std::size_t first{windows[0].first % n};
        std::size_t i{windows[0].first % n};
        for (const auto& window : windows) {
            for (std::size_t j = window.first; j < window.second; ++j, ++i) {
                values[i % n] = values[j % n];
            }
        }
        std::rotate(values.begin(), values.begin() + first, values.end());
        values.resize(m);
    }

    //! Count the number of \p values which have non-zero count.
    template<typename CONTAINER>
    static std::size_t countNotMissing(const CONTAINER& values) {
        return std::count_if(values.begin(), values.end(), [](const auto& value) {
            return CBasicStatistics::count(value) > 0.0;
        });
    }

private:
    static void removeComponents(const TSeasonalComponentVec& periods,
                                 const TMeanAccumulatorVecVec& components,
                                 TFloatMeanAccumulatorVec& values);
    static void checkForSeasonalDecomposition(const TDoubleVec& correlations,
                                              std::size_t maxCorrelationPeriod,
                                              double cutoff,
                                              std::size_t maxComponents,
                                              TSizeVec& candidatePeriods);
    static bool checkForTradingDayDecomposition(TFloatMeanAccumulatorVec& values,
                                                double outlierFraction,
                                                std::size_t day,
                                                std::size_t week,
                                                TSeasonalComponentVec& periods,
                                                TMeanAccumulatorVecVec& components,
                                                TSizeVec& candidatePeriods,
                                                TSeasonalComponentVec& result,
                                                TOptionalSize startOfWeekOverride,
                                                double significantPValue);
    template<typename VALUES, typename COMPONENT>
    static void doFitSeasonalComponents(const TSeasonalComponentVec& periods,
                                        const VALUES& values,
                                        std::vector<COMPONENT>& components);
    template<typename PREDICTOR, typename VALUES, typename COMPONENT>
    static void fitSeasonalComponentsMinusPrediction(const SSeasonalComponentSummary& period,
                                                     const PREDICTOR& predictor,
                                                     const VALUES& values,
                                                     COMPONENT& component);
    static double mean(const TFloatMeanAccumulator& value) {
        return CBasicStatistics::mean(value);
    }
    static double count(const TFloatMeanAccumulator& value) {
        return CBasicStatistics::count(value);
    }
};
}
}

#endif // INCLUDED_ml_maths_CSignal_h
