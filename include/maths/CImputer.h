/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CImputer_h
#define INCLUDED_ml_maths_CImputer_h

#include <maths/CKdTree.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

#include <cstddef>

namespace ml {
namespace maths {

//! \brief Implements methods for imputing missing data.
//!
//! DETAILS:\n
//! Currently, we implement random and a variety of nearest neighbour
//! approaches. See CImputer::impute for more details on the exact
//! algorithms. In addition, one can use regression relief to down
//! sample the most relevant attributes for imputing each missing
//! attribute.
//!
//! IMPLEMENTATION:\n
//! This uses ptrdiff_t throughout because Eigen expects a signed type
//! for enumerating vector coordinates (so that it can represent dynamic
//! with -1).
class MATHS_EXPORT CImputer {
public:
    using TIntDoublePr = std::pair<std::ptrdiff_t, double>;
    using TIntDoublePrVec = std::vector<TIntDoublePr>;
    using TIntDoublePrVecVec = std::vector<TIntDoublePrVec>;
    using TDenseVector = CDenseVector<double>;
    using TDenseVectorVec = std::vector<TDenseVector>;

    //! The method used to filter attributes.
    enum EFilterAttributes {
        E_NoFiltering, //!< No attribute filtering
        E_RReliefF     //!< Relief filtering of most relevant attributes.
    };

    //! The method used to inpute.
    enum EMethod {
        E_Random,             //!< Sample the attribute distribution.
        E_NNPlain,            //!< Weighted mean of nearest neighbours.
        E_NNBaggedSamples,    //!< Nearest neighbours with bagged samples.
        E_NNBaggedAttributes, //!< Nearest neighbours with bagged attributes.
        E_NNRandom            //!< Fully random nearest neighbours.
    };

public:
    //! The proportion of attributes to retain if filtering based on quality.
    static const double FILTER_FRACTION;
    //! The proportion of attributes to sample if using bagged attributes.
    static const double FRACTION_OF_VALUES_IN_BAG;
    //! The number of bags to use.
    static const std::size_t NUMBER_BAGS;
    //! The number of neighbours to use to impute if using these methods.
    static const std::size_t NUMBER_NEIGHBOURS;

public:
    explicit CImputer(double filterFraction = FILTER_FRACTION,
                      double fractionOfValuesInBag = FRACTION_OF_VALUES_IN_BAG,
                      std::size_t numberBags = NUMBER_BAGS,
                      std::size_t numberNeighbours = NUMBER_NEIGHBOURS);

    //! This implements a variety of methods for imputation depending
    //! on \p filter and \p method.
    //!
    //! The nearest neighbour methods use the weighted approach proposed
    //! by Troyanskaya et al in "Missing value estimation methods for DNA
    //! microarray": the weight is proportional to the inverse distance
    //! measure.
    //!
    //! There are three randomized methods used to avoid bias due to noisy
    //! poorly correlated samples and attributes (bagged samples, i.e.
    //! sample values with replacement, bagged attributes, i.e. sample
    //! attributes with replacement and both). Following the advice in
    //! https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4959387/ we don't
    //! expose the number of nearest neighbours as a configurable parameter
    //! and use a fixed small number.
    //!
    //! Optionally, we support filtering using the regression form of relief
    //! for each attribute in turn and select the best 20% of attributes to
    //! with which to impute missing values for that attribute. See the section
    //! 2.3 in http://lkm.fri.uni-lj.si/rmarko/papers/robnik03-mlj.pdf.
    //!
    //! \note IMPORTANT \p values should be canonical, i.e. the coordinates
    //! must be sorted in ascending order.
    //! \warning IMPORTANT \p values is destroyed by this method. This is
    //! because the result contains the same information and so we want to
    //! avoid duplicating state.
    void impute(EFilterAttributes filter,
                EMethod method,
                std::ptrdiff_t dimension,
                const TIntDoublePrVecVec& values,
                TDenseVectorVec& result) const;

private:
    using TIntVec = std::vector<std::ptrdiff_t>;
    using TIntVecVec = std::vector<TIntVec>;
    using TDenseVectorVecVec = std::vector<TDenseVectorVec>;

private:
    //! Find the vectors with no missing values from attributes in \p bag.
    //!
    //! If \p bag is empty then all attributes are considered.
    void donors(const TIntVec& attributeBag,
                std::ptrdiff_t dimension,
                const TIntDoublePrVecVec& values,
                TIntVec& donors) const;

    //! Find the attributes which are best for predicting the variables
    //! in \p target, using the regression form of relief to measure the
    //! attributes quality.
    void rReliefF(std::ptrdiff_t target, const TDenseVectorVec& donors, TIntVec& best) const;

    //! Generate the bags for imputing based on method.
    void baggedSample(EMethod method,
                      std::ptrdiff_t dimension,
                      const TDenseVectorVec& donors,
                      TDenseVectorVecVec& bags) const;

    //! Impute by randomly sampling from the empirical distribution
    //! of each attribute.
    void imputeRandom(std::ptrdiff_t dimension,
                      const TIntDoublePrVecVec& values,
                      const TIntVecVec& missing,
                      TDenseVectorVec& recipients) const;

    //! Impute the \p attribute of \p recipient using the mean of the
    //! nearest neighbours in \p bag.
    double imputeNearestNeighbour(std::ptrdiff_t attribute,
                                  const TIntDoublePrVec& recipient,
                                  const TDenseVectorVecVec& bags) const;

    //! Get the Euclidean distance between \p x and the projection of
    //! \p y onto coordinates of \p x.
    double distance(const TIntDoublePrVec& x, const TDenseVector& y) const;

    //! Convert a sparse vector representation to a dense one.
    TDenseVector toDense(std::ptrdiff_t dimension, const TIntDoublePrVec& sparse) const;

private:
    //! The random number generator.
    mutable CPRNG::CXorShift1024Mult m_Rng;
    //! The proportion of attributes retained if filter by quality.
    double m_FilterFraction;
    //! The fraction of values to sample for the bag.
    double m_FractionInBag;
    //! The number of bags to use.
    std::size_t m_NumberBags;
    //! The number of neighbours to use if using nearest neighbours.
    std::size_t m_NumberNeighbours;
};
}
}

#endif // INCLUDED_ml_maths_CImputer_h
