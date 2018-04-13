/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CNaiveBayes_h
#define INCLUDED_ml_maths_CNaiveBayes_h

#include <maths/ImportExport.h>

#include <maths/CPrior.h>

#include <boost/optional.hpp>
#include <boost/unordered_map.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
struct SDistributionRestoreParams;

//! \brief The interface expected by CNaiveBayes for implementations
//! of the class conditional density functions.
class MATHS_EXPORT CNaiveBayesFeatureDensity {
public:
    using TDouble1Vec = core::CSmallVector<double, 1>;

public:
    virtual ~CNaiveBayesFeatureDensity() = default;

    //! Create and return a clone.
    //!
    //! \note The caller owns this.
    virtual CNaiveBayesFeatureDensity* clone() const = 0;

    //! Initialize by reading state from \p traverser.
    virtual bool acceptRestoreTraverser(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser) = 0;

    //! Persist state by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

    //! Set the data type.
    virtual void dataType(maths_t::EDataType dataType) = 0;

    //! Add the value \p x.
    virtual void add(const TDouble1Vec& x) = 0;

    //! Compute the log value of the density function at \p x.
    virtual double logValue(const TDouble1Vec& x) const = 0;

    //! Compute the density at the mode.
    virtual double logMaximumValue() const = 0;

    //! Age out old values density to account for \p time passing.
    virtual void propagateForwardsByTime(double time) = 0;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Get the static size of this object.
    virtual std::size_t staticSize() const = 0;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const = 0;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed) const = 0;

    //! Get a human readable description of the class density function.
    virtual std::string print() const = 0;
};

//! \brief An implementation of the class conditional density function
//! based on the CPrior hierarchy.
class MATHS_EXPORT CNaiveBayesFeatureDensityFromPrior final : public CNaiveBayesFeatureDensity {
public:
    CNaiveBayesFeatureDensityFromPrior() = default;
    CNaiveBayesFeatureDensityFromPrior(const CPrior& prior);

    //! Create and return a clone.
    //!
    //! \note The caller owns this.
    virtual CNaiveBayesFeatureDensityFromPrior* clone() const;

    //! Initialize by reading state from \p traverser.
    virtual bool acceptRestoreTraverser(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Add the value \p x.
    virtual void add(const TDouble1Vec& x);

    //! Compute the log value of the density function at \p x.
    virtual double logValue(const TDouble1Vec& x) const;

    //! Compute the density at the mode.
    virtual double logMaximumValue() const;

    //! Set the data type.
    virtual void dataType(maths_t::EDataType dataType);

    //! Age out old values density to account for \p time passing.
    virtual void propagateForwardsByTime(double time);

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the static size of this object.
    virtual std::size_t staticSize() const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed) const;

    //! Get a human readable description of the class density function.
    virtual std::string print() const;

private:
    using TPriorPtr = boost::shared_ptr<CPrior>;

private:
    //! The density model.
    TPriorPtr m_Prior;
};

//! \brief Implements a Naive Bayes classifier.
class MATHS_EXPORT CNaiveBayes {
public:
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TDoubleSizePrVec = std::vector<TDoubleSizePr>;
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble1VecVec = std::vector<TDouble1Vec>;
    using TOptionalDouble = boost::optional<double>;

public:
    explicit CNaiveBayes(const CNaiveBayesFeatureDensity& exemplar,
                         double decayRate = 0.0,
                         TOptionalDouble minMaxLogLikelihoodToUseFeature = TOptionalDouble());
    CNaiveBayes(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Efficiently swap the contents of this and \p other.
    void swap(CNaiveBayes& other);

    //! Check if any training data has been added initialized.
    bool initialized() const;

    //! This can be used to optionally seed the class counts
    //! with \p counts. These are added on to data class counts
    //! to compute the class posterior probabilities.
    void initialClassCounts(const TDoubleSizePrVec& counts);

    //! Add a training data point comprising the pair \f$(x,l)\f$
    //! for feature vector \f$x\f$ and class label \f$l\f$.
    //!
    //! \param[in] label The class label for \p x.
    //! \param[in] x The feature values.
    //! \note \p x size should be equal to the number of features.
    //! A feature is missing is indicated by passing an empty vector
    //! for that feature.
    void addTrainingDataPoint(std::size_t label, const TDouble1VecVec& x);

    //! Set the data type.
    void dataType(maths_t::EDataType dataType);

    //! Age out old values from the class conditional densities
    //! to account for \p time passing.
    void propagateForwardsByTime(double time);

    //! Get the top \p n class probabilities for \p x.
    //!
    //! \param[in] n The number of class probabilities to estimate.
    //! \param[in] x The feature values.
    //! \note \p x size should be equal to the number of features.
    //! A feature is missing is indicated by passing an empty vector
    //! for that feature.
    TDoubleSizePrVec highestClassProbabilities(std::size_t n, const TDouble1VecVec& x) const;

    //! Get the probability of the class labeled \p label for \p x.
    //!
    //! \param[in] label The label of the class of interest.
    //! \param[in] x The feature values.
    //! \note \p x size should be equal to the number of features.
    //! A feature is missing is indicated by passing an empty vector
    //! for that feature.
    double classProbability(std::size_t label, const TDouble1VecVec& x) const;

    //! Get the probabilities of all the classes for \p x.
    //!
    //! \param[in] x The feature values.
    //! \note \p x size should be equal to the number of features.
    //! A feature is missing is indicated by passing an empty vector
    //! for that feature.
    TDoubleSizePrVec classProbabilities(const TDouble1VecVec& x) const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed = 0) const;

    //! Get a human readable description of the classifier.
    std::string print() const;

private:
    using TFeatureDensityPtr = boost::shared_ptr<CNaiveBayesFeatureDensity>;
    using TFeatureDensityPtrVec = std::vector<TFeatureDensityPtr>;

    //! \brief The data associated with a class.
    struct SClass {
        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser);
        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;
        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;
        //! Get the memory used by this object.
        std::size_t memoryUsage() const;
        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! The number of examples in this class.
        double s_Count = 0.0;
        //! The feature conditional densities for this class.
        TFeatureDensityPtrVec s_ConditionalDensities;
    };

    using TSizeClassUMap = boost::unordered_map<std::size_t, SClass>;

private:
    //! Initialize by reading state from \p traverser.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser);

    //! Validate \p x.
    bool validate(const TDouble1VecVec& x) const;

private:
    //! It is not always appropriate to use features with very low
    //! probability in all classes to discriminate: the class choice
    //! will be very sensitive to the underlying conditional density
    //! model. This is a cutoff (for the minimum maximum class log
    //! likelihood) in order to use a feature.
    TOptionalDouble m_MinMaxLogLikelihoodToUseFeature;

    //! Controls the rate at which data are aged out.
    double m_DecayRate;

    //! An exemplar for creating conditional densities.
    TFeatureDensityPtr m_Exemplar;

    //! The class conditional density estimates and weights.
    TSizeClassUMap m_ClassConditionalDensities;
};
}
}

#endif // INCLUDED_ml_maths_CNaiveBayes_h
