/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CImputer.h>

#include <core/CMaskIterator.h>

#include <maths/CBasicStatistics.h>
#include <maths/COrderings.h>
#include <maths/CSampling.h>
#include <maths/CSetTools.h>

#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TIntVec = std::vector<std::ptrdiff_t>;
const TIntVec EMPTY_BAG;
}

CImputer::CImputer(double filterFraction, double fractionInBag, std::size_t numberBags, std::size_t numberNeighbours)
    : m_FilterFraction(filterFraction), m_FractionInBag(fractionInBag),
      m_NumberBags(numberBags), m_NumberNeighbours(numberNeighbours) {
}

void CImputer::impute(EFilterAttributes filter,
                      EMethod method,
                      std::ptrdiff_t dimension,
                      const TIntDoublePrVecVec& values,
                      TDenseVectorVec& result) const {
    result.clear();

    if (values.empty()) {
        return;
    }

    // Extract donor values, i.e. values with no missing attributes.
    TIntVec donors_;
    this->donors(EMPTY_BAG, dimension, values, donors_);

    // Get the complement of the donors, i.e. values with at least one
    // missing attribute.
    std::ptrdiff_t n{static_cast<std::ptrdiff_t>(values.size())};
    TIntVec recipients_;
    recipients_.reserve(values.size() - donors_.size());
    std::set_difference(boost::iterators::make_counting_iterator(std::ptrdiff_t(0)),
                        boost::iterators::make_counting_iterator(n), donors_.begin(),
                        donors_.end(), std::back_inserter(recipients_));

    // Extract the indices of all the missing attribute values.
    TIntVecVec missing(dimension);
    for (std::ptrdiff_t attribute = 0; attribute < dimension; ++attribute) {
        for (std::size_t i = 0u; i < recipients_.size(); ++i) {
            const TIntDoublePrVec& value{values[recipients_[i]]};
            if (!std::binary_search(value.begin(), value.end(), attribute,
                                    COrderings::SFirstLess())) {
                missing[attribute].push_back(static_cast<std::ptrdiff_t>(i));
            }
        }
    }

    if (recipients_.size() == 0) {
        result.reserve(values.size());
        for (const auto& value : values) {
            result.push_back(this->toDense(dimension, value));
        }
    } else {
        result.resize(values.size());

        TDenseVectorVec recipients;
        recipients.reserve(recipients_.size());
        for (auto i : recipients_) {
            recipients.push_back(this->toDense(dimension, values[i]));
        }

        switch (method) {
        case E_Random:
            this->imputeRandom(dimension, values, missing, recipients);
            for (auto i : donors_) {
                result[i] = this->toDense(dimension, values[i]);
            }
            break;

        case E_NNPlain:
        case E_NNBaggedSamples:
        case E_NNBaggedAttributes:
        case E_NNRandom: {
            TDenseVectorVec donors;
            donors.reserve(donors_.size());
            for (auto i : donors_) {
                donors.push_back(this->toDense(dimension, values[i]));
            }

            // We need to get a list of suitable attributes for each attribute
            // to impute since regression relief depends on the attribute to be
            // imputed.
            switch (filter) {
            case E_NoFiltering:
                break;
            case E_RReliefF: /*TODO*/
                break;
            }

            // Filled in with the bags of donors used to impute.
            TDenseVectorVecVec bags;
            this->baggedSample(method, dimension, donors, bags);

            // Impute missing values for each attribute in turn.
            for (std::ptrdiff_t attribute = 0; attribute < dimension; ++attribute) {
                for (auto i : missing[attribute]) {
                    recipients[i][attribute] = this->imputeNearestNeighbour(
                        attribute, values[recipients_[i]], bags);
                }
            }
            for (std::size_t i = 0u; i < donors_.size(); ++i) {
                result[donors_[i]] = std::move(donors[i]);
            }
            break;
        }
        }

        for (std::size_t i = 0u; i < recipients_.size(); ++i) {
            result[recipients_[i]] = std::move(recipients[i]);
        }
    }
}

void CImputer::donors(const TIntVec& attributeBag_,
                      std::ptrdiff_t dimension,
                      const TIntDoublePrVecVec& values,
                      TIntVec& donors) const {
    donors.clear();

    if (attributeBag_.empty()) {
        for (std::size_t i = 0u; i < values.size(); ++i) {
            if (values[i].size() == static_cast<std::size_t>(dimension)) {
                donors.push_back(i);
            }
        }
    } else {
        TIntVec attributeBag(attributeBag_);
        std::sort(attributeBag.begin(), attributeBag.end());
        for (std::size_t i = 0u; i < values.size(); ++i) {
            if (std::equal(attributeBag.begin(), attributeBag.end(), values[i].begin(),
                           [](std::ptrdiff_t lhs, const TIntDoublePr& rhs) {
                               return lhs == rhs.first;
                           })) {
                donors.push_back(i);
            }
        }
    }
}

void CImputer::rReliefF(std::ptrdiff_t /*target*/,
                        const TDenseVectorVec& /*donors*/,
                        TIntVec& /*best*/) const {
    // TODO
}

void CImputer::baggedSample(EMethod method,
                            std::ptrdiff_t dimension,
                            const TDenseVectorVec& donors,
                            TDenseVectorVecVec& bags) const {
    std::ptrdiff_t m{static_cast<std::ptrdiff_t>(
        std::max(m_FractionInBag * static_cast<double>(donors.size()) + 0.5, 1.0))};
    std::ptrdiff_t n{
        static_cast<std::ptrdiff_t>(::sqrt(static_cast<double>(dimension)) + 0.5)};

    TIntVec sampleBag;
    TIntVec attributeBag;
    TDenseVectorVec projected(donors.size(), TDenseVector(n));
    switch (method) {
    case E_Random:
        break;
    case E_NNPlain:
        bags.push_back(donors);
        break;
    case E_NNBaggedSamples:
        bags.resize(m_NumberBags);
        for (std::size_t b = 0u; b < m_NumberBags; ++b) {
            CSampling::uniformSample(
                m_Rng, 0, static_cast<std::ptrdiff_t>(donors.size()), m, sampleBag);
            bags[b].assign(core::begin_masked(donors, sampleBag),
                           core::end_masked(donors, sampleBag));
        }
        break;
    case E_NNBaggedAttributes:
        bags.resize(m_NumberBags);
        for (std::size_t b = 0u; b < m_NumberBags; ++b) {
            CSampling::uniformSample(m_Rng, 0, dimension, n, attributeBag);
            for (std::size_t i = 0u; i < donors.size(); ++i) {
                for (std::size_t j = 0u; j < attributeBag.size(); ++j) {
                    projected[i][j] = donors[i][attributeBag[j]];
                }
            }
            bags[b] = projected;
        }
        break;
    case E_NNRandom:
        bags.resize(m_NumberBags);
        for (std::size_t b = 0u; b < m_NumberBags; ++b) {
            CSampling::uniformSample(m_Rng, 0, dimension, n, attributeBag);
            for (std::size_t i = 0u; i < donors.size(); ++i) {
                for (std::size_t j = 0u; j < attributeBag.size(); ++j) {
                    projected[i][j] = donors[i][attributeBag[j]];
                }
            }
            CSampling::uniformSample(
                m_Rng, 0, static_cast<std::ptrdiff_t>(donors.size()), m, sampleBag);
            bags[b].assign(core::begin_masked(projected, sampleBag),
                           core::end_masked(projected, sampleBag));
        }
        break;
    }
}

void CImputer::imputeRandom(std::ptrdiff_t dimension,
                            const TIntDoublePrVecVec& values,
                            const TIntVecVec& missing,
                            TDenseVectorVec& recipients) const {
    TDoubleVec support;
    for (std::ptrdiff_t attribute = 0; attribute < dimension; ++attribute) {
        // Find the observed values for this attribute.
        support.clear();
        for (const auto& value : values) {
            auto attribute_ = std::lower_bound(value.begin(), value.end(), attribute,
                                               COrderings::SFirstLess());
            if (attribute_ != value.end() && attribute_->first == attribute) {
                support.push_back(attribute_->second);
            }
        }

        for (std::size_t i = 0u; i < missing[attribute].size(); ++i) {
            recipients[missing[attribute][i]][attribute] =
                support[CSampling::uniformSample(m_Rng, 0, support.size())];
        }
    }
}

double CImputer::imputeNearestNeighbour(std::ptrdiff_t attribute,
                                        const TIntDoublePrVec& recipient,
                                        const TDenseVectorVecVec& bags) const {
    using TDoubleDoublePr = std::pair<double, double>;
    using TMinAccumulator = CBasicStatistics::COrderStatisticsHeap<TDoubleDoublePr>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    TMeanAccumulator result;

    TMinAccumulator neighbours(m_NumberNeighbours);
    for (const auto& bag : bags) {
        for (const auto& donor : bag) {
            neighbours.add({distance(recipient, donor), donor(attribute)});
        }

        // Trap the case that the distance is zero in which case we'll simply
        // use a relatively large but finite maximum weight for those samples.
        double min{0.0};
        for (const auto& neighbour : neighbours) {
            min = std::max(min, neighbour.first);
        }
        min = min == 0.0 ? 1.0 : min / 1000.0;

        for (const auto& neighbour : neighbours) {
            double weight{1.0 / std::max(neighbour.first, min)};
            result.add(neighbour.second, weight);
        }
    }

    return CBasicStatistics::mean(result);
}

double CImputer::distance(const TIntDoublePrVec& x, const TDenseVector& y) const {
    double result{0.0};
    for (const auto& coordinate : x) {
        result += ::pow(coordinate.second - y(coordinate.first), 2.0);
    }
    return ::sqrt(result);
}

CImputer::TDenseVector CImputer::toDense(std::ptrdiff_t dimension,
                                         const TIntDoublePrVec& sparse) const {
    TDenseVector result(dimension);
    for (const auto& attribute : sparse) {
        result[attribute.first] = attribute.second;
    }
    return result;
}

const double CImputer::FILTER_FRACTION{0.2};
const std::size_t CImputer::NUMBER_NEIGHBOURS{3};
const double CImputer::FRACTION_OF_VALUES_IN_BAG{0.1};
const std::size_t CImputer::NUMBER_BAGS{20};
}
}
