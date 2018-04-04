/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CRandomProjectionClusterer_h
#define INCLUDED_ml_maths_CRandomProjectionClusterer_h

#include <maths/CAgglomerativeClusterer.h>
#include <maths/CBasicStatistics.h>
#include <maths/CGramSchmidt.h>
#include <maths/CKMeansFast.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CNaturalBreaksClassifier.h>
#include <maths/CSampling.h>
#include <maths/CXMeans.h>

#include <boost/array.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ml {
namespace maths {

//! \brief Common functionality for random projection clustering.
//!
//! DESCRIPTION:\n
//! This implements the core functionality for clustering via
//! random projections.
//!
//! The idea is to construct a set of subspaces of low dimension
//! by projecting the data points onto an orthogonalisation of
//! randomly generated vectors. Specifically, this generates a
//! collection of random vectors \f$[x]_i ~ N(0,1)\f$ and then
//! constructs an orthonormal basis by the Gram-Schmidt process.
//!
//! Having generated a number of different random projections
//! of the data, this measures the similarity of the i'th and
//! j'th point by looking at the average probability they belong
//! to the same cluster over the ensemble of clusterings. This
//! step is achieved by associating a generative model, specifically,
//! a weighted mixture of normals with each clustering.
//!
//! Finally, hierarchical agglomerative clustering is performed
//! on the resulting similarity matrix together with model selection
//! to choose the number of clusters.
//!
//! For more details see http://people.ee.duke.edu/~lcarin/random-projection-for-high.pdf
template<std::size_t N>
class CRandomProjectionClusterer {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;

public:
    virtual ~CRandomProjectionClusterer() = default;

    //! Set up the projections.
    virtual bool initialise(std::size_t numberProjections, std::size_t dimension) {
        m_Dimension = dimension;
        if (!this->generateProjections(numberProjections)) {
            LOG_ERROR("Failed to generate projections");
            return false;
        }
        return true;
    }

protected:
    using TVector = CVector<double>;
    using TVectorVec = std::vector<TVector>;
    using TVectorArray = boost::array<TVector, N>;
    using TVectorArrayVec = std::vector<TVectorArray>;

protected:
    //! Get the random number generator.
    CPRNG::CXorShift1024Mult& rng() const { return m_Rng; }

    //! Get the projections.
    const TVectorArrayVec& projections() const { return m_Projections; }

    //! Generate \p b random projections.
    bool generateProjections(std::size_t b) {
        m_Projections.clear();

        if (b == 0) {
            return true;
        }

        if (m_Dimension <= N) {
            m_Projections.resize(1);
            TVectorArray& projection = m_Projections[0];
            for (std::size_t i = 0u; i < N; ++i) {
                projection[i].extend(m_Dimension, 0.0);
                if (i < m_Dimension) {
                    projection[i](i) = 1.0;
                }
            }
            return true;
        }

        m_Projections.resize(b);

        TDoubleVec components;
        CSampling::normalSample(m_Rng, 0.0, 1.0, b * N * m_Dimension, components);
        for (std::size_t i = 0u; i < b; ++i) {
            TVectorArray& projection = m_Projections[i];
            for (std::size_t j = 0u; j < N; ++j) {
                projection[j].assign(&components[(i * N + j) * m_Dimension], &components[(i * N + j + 1) * m_Dimension]);
            }

            if (!CGramSchmidt::basis(projection)) {
                LOG_ERROR("Failed to construct basis");
                return false;
            }
        }

        return true;
    }

    //! Extend the projections for an increase in data
    //! dimension to \p dimension.
    bool extendProjections(std::size_t dimension) {
        using TDoubleVecArray = boost::array<TDoubleVec, N>;

        if (dimension <= m_Dimension) {
            return true;
        } else if (dimension <= N) {
            TVectorArray& projection = m_Projections[0];
            for (std::size_t i = m_Dimension; i < dimension; ++i) {
                projection[i](i) = 1.0;
            }
            return true;
        }

        std::size_t b = m_Projections.size();
        std::size_t d = dimension - m_Dimension;
        double alpha = static_cast<double>(m_Dimension) / static_cast<double>(dimension);
        double beta = 1.0 - alpha;

        TDoubleVecArray extension;
        TDoubleVec components;
        CSampling::normalSample(m_Rng, 0.0, 1.0, b * N * d, components);
        for (std::size_t i = 0u; i < b; ++i) {
            for (std::size_t j = 0u; j < N; ++j) {
                extension[j].assign(&components[(i * N + j) * d], &components[(i * N + j + 1) * d]);
            }

            if (!CGramSchmidt::basis(extension)) {
                LOG_ERROR("Failed to construct basis");
                return false;
            }

            for (std::size_t j = 0u; j < N; ++j) {
                scale(extension[j], beta);
                TVector& projection = m_Projections[i][j];
                projection *= alpha;
                projection.reserve(dimension);
                projection.extend(extension[j].begin(), extension[j].end());
            }
        }

        return true;
    }

private:
    //! Scale the values in the vector \p x by \p scale.
    void scale(TDoubleVec& x, double scale) {
        for (std::size_t i = 0u; i < x.size(); ++i) {
            x[i] *= scale;
        }
    }

private:
    //! The random number generator.
    mutable CPRNG::CXorShift1024Mult m_Rng;

    //! The dimension of the data to project.
    std::size_t m_Dimension;

    //! The projections.
    TVectorArrayVec m_Projections;
};

//! \brief Implements random projection clustering for batches
//! of data points.
template<std::size_t N>
class CRandomProjectionClustererBatch : public CRandomProjectionClusterer<N> {
public:
    using TDoubleVec = typename CRandomProjectionClusterer<N>::TDoubleVec;
    using TSizeVec = typename CRandomProjectionClusterer<N>::TSizeVec;
    using TVector = typename CRandomProjectionClusterer<N>::TVector;
    using TVectorVec = typename CRandomProjectionClusterer<N>::TVectorVec;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TSizeUSet = boost::unordered_set<std::size_t>;
    using TVectorNx1 = CVectorNx1<double, N>;
    using TEigenVectorNx1 = typename SDenseVector<TVectorNx1>::Type;
    using TVectorNx1Vec = std::vector<TVectorNx1>;
    using TVectorNx1VecVec = std::vector<TVectorNx1Vec>;
    using TSymmetricMatrixNxN = CSymmetricMatrixNxN<double, N>;
    using TSvdNxN = Eigen::JacobiSVD<typename SDenseMatrix<TSymmetricMatrixNxN>::Type>;
    using TSvdNxNVec = std::vector<TSvdNxN>;
    using TSvdNxNVecVec = std::vector<TSvdNxNVec>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;

public:
    CRandomProjectionClustererBatch(double compression) : m_Compression(compression) {}

    virtual ~CRandomProjectionClustererBatch() = default;

    //! Create the \p numberProjections random projections.
    //!
    //! \param[in] numberProjections The number of projections
    //! to create.
    //! \param[in] dimension The dimension of the space to project.
    virtual bool initialise(std::size_t numberProjections, std::size_t dimension) {
        m_ProjectedData.resize(numberProjections);
        return this->CRandomProjectionClusterer<N>::initialise(numberProjections, dimension);
    }

    //! Reserve space for \p n data points.
    void reserve(std::size_t n) {
        for (std::size_t i = 0u; i < m_ProjectedData.size(); ++i) {
            m_ProjectedData[i].reserve(n);
        }
    }

    //! Add projected data for \p x.
    void add(const TVector& x) {
        for (std::size_t i = 0u; i < this->projections().size(); ++i) {
            TVectorNx1 px;
            for (std::size_t j = 0u; j < N; ++j) {
                px(j) = this->projections()[i][j].inner(x);
            }
            m_ProjectedData[i].push_back(px);
        }
    }

    //! Compute the clusters.
    //!
    //! \param[in] clusterer The object responsible for clustering
    //! the projected data points.
    //! \param[in] result Filled in with the final agglomerative
    //! clustering of the different projections.
    template<typename CLUSTERER>
    void run(CLUSTERER clusterer, TSizeVecVec& result) const {
        if (m_ProjectedData.empty()) {
            return;
        }

        std::size_t b = m_ProjectedData.size();

        // Filled in with the weights of the clusterings.
        TDoubleVecVec W(b);
        // Filled in with the sample means of the clusterings.
        TVectorNx1VecVec M(b);
        // Filled in with the SVDs of the sample covariances
        // of the clusterings.
        TSvdNxNVecVec C(b);
        // Filled in with the sample points indices.
        TSizeUSet I;

        // Compute the projected clusterings and sampling.
        this->clusterProjections(clusterer, W, M, C, I);

        // Compute the sample neighbourhoods.
        std::size_t h = I.size();
        TSizeVecVec H(h);
        this->neighbourhoods(I, H);

        // Compute the cluster similarities.
        TDoubleVecVec S(h);
        this->similarities(W, M, C, H, S);

        // Run agglomerative clustering and choose number of clusters.
        this->clusterNeighbourhoods(S, H, result);
    }

protected:
    //! \brief Hashes a vector.
    struct SHashVector {
        template<typename VECTOR>
        std::size_t operator()(const VECTOR& lhs) const {
            return static_cast<std::size_t>(boost::unwrap_ref(lhs).checksum());
        }
    };
    //! \brief Checks two vectors for equality.
    struct SVectorsEqual {
        template<typename VECTOR>
        bool operator()(const VECTOR& lhs, const VECTOR& rhs) const {
            return boost::unwrap_ref(lhs) == boost::unwrap_ref(rhs);
        }
    };

protected:
    //! Compute the projected clusterings and find a good sampling
    //! of the points on which to perform agglomerative clustering.
    //!
    //! \param[in] clusterer The object responsible for clustering
    //! the projected data points.
    //! \param[out] W Filled in with the cluster weights.
    //! \param[out] M Filled in with the cluster sample means.
    //! \param[out] C Filled in with the SVD of cluster sample
    //! covariance matrices.
    //! \param[out] I Filled in with the indices of distinct sampled
    //! points.
    template<typename CLUSTERER>
    void clusterProjections(CLUSTERER clusterer, TDoubleVecVec& W, TVectorNx1VecVec& M, TSvdNxNVecVec& C, TSizeUSet& I) const {
        using TVectorNx1CRef = boost::reference_wrapper<const TVectorNx1>;
        using TVectorNx1CRefSizeUMap = boost::unordered_map<TVectorNx1CRef, std::size_t, SHashVector, SVectorsEqual>;
        using TClusterVec = typename CLUSTERER::TClusterVec;
        using TSampleCovariancesNxN = CBasicStatistics::SSampleCovariances<double, N>;

        std::size_t b = m_ProjectedData.size();
        std::size_t n = m_ProjectedData[0].size();

        // An index lookup for some projected points.
        TVectorNx1CRefSizeUMap lookup(n);
        // A placeholder for copy of i'th projected data.
        TVectorNx1Vec P;
        // Filled in with the probabilities of sampling the points (i,j)'th
        // cluster.
        TDoubleVec pij;
        // Filled in with a mapping from the candidates for sampling to the
        // actual points in the (i,j)'th cluster.
        TSizeVec fij;
        // Filled in with the samples of the (i,j)'th cluster.
        TSizeVec sij;

        for (std::size_t i = 0u; i < b; ++i) {
            LOG_TRACE("projection " << i);
            P = m_ProjectedData[i];

            // Create a lookup of points to their indices.
            lookup.clear();
            lookup.rehash(P.size());
            for (std::size_t j = 0u; j < m_ProjectedData[i].size(); ++j) {
                lookup[boost::cref(m_ProjectedData[i][j])] = j;
            }

            // Cluster the i'th projection.
            clusterer.setPoints(P);
            clusterer.run();
            const TClusterVec& clusters = clusterer.clusters();
            double ni = static_cast<double>(clusters.size());
            LOG_TRACE("# clusters = " << ni);

            for (std::size_t j = 0u; j < clusters.size(); ++j) {
                const TVectorNx1Vec& points = clusters[j].points();
                LOG_TRACE("# points = " << points.size());

                // Compute the number of points to sample from this cluster.
                std::size_t nij = points.size();
                double wij = static_cast<double>(nij) / static_cast<double>(n);
                std::size_t nsij = static_cast<std::size_t>(std::max(m_Compression * wij * ni, 1.0));
                LOG_TRACE("wij = " << wij << ", nsij = " << nsij);

                // Compute the cluster sample mean and covariance matrix.
                TSampleCovariancesNxN covariances;
                covariances.add(points);
                TVectorNx1 mij = CBasicStatistics::mean(covariances);
                TSvdNxN Cij(toDenseMatrix(CBasicStatistics::covariances(covariances)), Eigen::ComputeFullU | Eigen::ComputeFullV);

                // Compute the probability that a sample from the cluster
                // is a given point in the cluster.
                pij.clear();
                fij.clear();
                pij.reserve(nij);
                fij.reserve(nij);
                double pmax = boost::numeric::bounds<double>::lowest();
                for (std::size_t k = 0u; k < nij; ++k) {
                    std::size_t index = lookup[boost::cref(points[k])];
                    if (I.count(index) == 0) {
                        TEigenVectorNx1 x = toDenseVector(points[k] - mij);
                        pij.push_back(-0.5 * x.transpose() * Cij.solve(x));
                        fij.push_back(index);
                        pmax = std::max(pmax, pij.back());
                    }
                }

                if (pij.size() > 0) {
                    double Zij = 0.0;
                    for (std::size_t k = 0u; k < pij.size(); ++k) {
                        pij[k] = std::exp(pij[k] - pmax);
                        Zij += pij[k];
                    }
                    for (std::size_t k = 0u; k < pij.size(); ++k) {
                        pij[k] /= Zij;
                    }
                    LOG_TRACE("pij = " << core::CContainerPrinter::print(pij));

                    // Sample the cluster.
                    CSampling::categoricalSampleWithoutReplacement(this->rng(), pij, nsij, sij);
                    LOG_TRACE("sij = " << core::CContainerPrinter::print(sij));

                    // Save the relevant data for the i'th clustering.
                    for (std::size_t k = 0u; k < nsij; ++k) {
                        I.insert(fij[sij[k]]);
                    }
                }
                W[i].push_back(wij);
                M[i].push_back(mij);
                C[i].push_back(Cij);
            }
        }
    }

    //! Construct the neighbourhoods of each of the sampled points.
    //!
    //! \param[in] I The indices of distinct sampled points.
    //! \param[out] H Filled in with the neighbourhoods of each
    //! point in \p I, i.e. the indices of the closest points.
    void neighbourhoods(const TSizeUSet& I, TSizeVecVec& H) const {
        using TVectorSizeUMap = boost::unordered_map<TVector, std::size_t, SHashVector>;

        LOG_TRACE("I = " << core::CContainerPrinter::print(I));
        std::size_t b = m_ProjectedData.size();
        std::size_t n = m_ProjectedData[0].size();

        // Create a k-d tree of the sampled data points.
        TVectorVec S;
        S.reserve(I.size());
        TVector concat(b * N);
        for (auto i : I) {
            for (std::size_t j = 0u; j < b; ++j) {
                for (std::size_t k = 0u; k < N; ++k) {
                    concat(N * j + k) = m_ProjectedData[j][i](k);
                }
            }
            LOG_TRACE("concat = " << concat);
            S.push_back(concat);
        }
        TVectorSizeUMap lookup(S.size());
        for (std::size_t i = 0u; i < S.size(); ++i) {
            lookup[S[i]] = i;
        }
        CKdTree<TVector> samples;
        samples.build(S);

        // Compute the neighbourhoods.
        for (std::size_t i = 0u; i < n; ++i) {
            for (std::size_t j = 0u; j < b; ++j) {
                for (std::size_t k = 0u; k < N; ++k) {
                    concat(N * j + k) = m_ProjectedData[j][i](k);
                }
            }
            const TVector* nn = samples.nearestNeighbour(concat);
            if (!nn) {
                LOG_ERROR("No nearest neighbour of " << concat);
                continue;
            }
            LOG_TRACE("nn = " << *nn);
            H[lookup[*nn]].push_back(i);
        }
        LOG_TRACE("H = " << core::CContainerPrinter::print(H));
    }

    //! Compute the similarities between neighbourhoods.
    //!
    //! \param[in] W The cluster weights.
    //! \param[in] M The cluster sample means.
    //! \param[in] C The SVD of cluster sample covariance matrices.
    //! \param[in] H The neighbourhoods of each point in \p I,
    //! i.e. the indices of the closest points.
    //! \param[out] S Filled in with the mean similarities between
    //! neighbourhoods over the different clusterings.
    void
    similarities(const TDoubleVecVec& W, const TVectorNx1VecVec& M, const TSvdNxNVecVec& C, const TSizeVecVec& H, TDoubleVecVec& S) const {
        std::size_t b = m_ProjectedData.size();
        std::size_t h = H.size();

        TMeanAccumulatorVecVec S_(h);

        TVectorVec Pi(h);
        for (std::size_t i = 0u; i < b; ++i) {
            const TVectorNx1Vec& X = m_ProjectedData[i];
            const TDoubleVec& Wi = W[i];
            const TVectorNx1Vec& Mi = M[i];
            const TSvdNxNVec& Ci = C[i];
            LOG_TRACE("W(i) = " << core::CContainerPrinter::print(Wi));
            LOG_TRACE("M(i) = " << core::CContainerPrinter::print(Mi));

            std::size_t nci = Mi.size();
            std::fill_n(Pi.begin(), h, TVector(nci));

            // Compute the probability each neighbourhood is from
            // a given cluster.
            for (std::size_t c = 0u; c < nci; ++c) {
                double wic = std::log(Wi[c]) - 0.5 * this->logDeterminant(Ci[c]);
                LOG_TRACE("  w(" << i << "," << c << ") = " << wic);
                for (std::size_t j = 0u; j < h; ++j) {
                    std::size_t hj = H[j].size();
                    Pi[j](c) = static_cast<double>(hj) * wic;
                    for (std::size_t k = 0u; k < hj; ++k) {
                        TEigenVectorNx1 x = toDenseVector(X[H[j][k]] - Mi[c]);
                        Pi[j](c) -= 0.5 * x.transpose() * Ci[c].solve(x);
                    }
                    LOG_TRACE("    P(" << j << "," << c << ") = " << Pi[j](c));
                }
            }
            for (std::size_t j = 0u; j < h; ++j) {
                double Pmax = *std::max_element(Pi[j].begin(), Pi[j].end());
                double Z = 0.0;
                for (std::size_t c = 0u; c < nci; ++c) {
                    Pi[j](c) = std::exp(Pi[j](c) - Pmax);
                    Z += Pi[j](c);
                }
                for (std::size_t c = 0u; c < nci; ++c) {
                    Pi[j](c) /= Z;
                }
                LOG_TRACE("  P(" << j << ") = " << Pi[j]);
            }

            // Compute the similarities.
            for (std::size_t j = 0u; j < h; ++j) {
                S_[j].resize(j + 1);
                for (std::size_t k = 0u; k <= j; ++k) {
                    S_[j][k].add(-std::log(std::max(Pi[j].inner(Pi[k]), boost::numeric::bounds<double>::smallest())));
                }
            }
        }
        for (std::size_t i = 0u; i < S_.size(); ++i) {
            S[i].reserve(S_[i].size());
            for (std::size_t j = 0u; j < S_[i].size(); ++j) {
                S[i].push_back(CBasicStatistics::mean(S_[i][j]));
            }
        }
    }

    //! Extract the clustering of the neighbourhoods based on
    //! their similarities.
    //!
    //! \param[in] S The similarities between neighbourhoods.
    //! \param[in] H The neighbourhoods.
    //! \param[out] result Filled in with the clustering of the
    //! underlying points.
    void clusterNeighbourhoods(TDoubleVecVec& S, const TSizeVecVec& H, TSizeVecVec& result) const {
        using TNode = CAgglomerativeClusterer::CNode;
        using TDoubleTuple = CNaturalBreaksClassifier::TDoubleTuple;
        using TDoubleTupleVec = CNaturalBreaksClassifier::TDoubleTupleVec;

        result.clear();

        CAgglomerativeClusterer agglomerative;
        agglomerative.initialize(S);
        CAgglomerativeClusterer::TNodeVec tree;
        agglomerative.run(CAgglomerativeClusterer::E_Average, tree);

        TDoubleTupleVec heights;
        heights.reserve(tree.size());
        for (std::size_t i = 0u; i < tree.size(); ++i) {
            heights.push_back(TDoubleTuple());
            heights.back().add(tree[i].height());
        }
        LOG_TRACE("heights = " << core::CContainerPrinter::print(heights));

        TSizeVec splits;
        if (CNaturalBreaksClassifier::naturalBreaks(heights,
                                                    2, // Number splits
                                                    0, // Minimum cluster size
                                                    CNaturalBreaksClassifier::E_TargetDeviation,
                                                    splits)) {
            double height = CBasicStatistics::mean(heights[splits[0] - 1]);
            LOG_TRACE("split = " << core::CContainerPrinter::print(splits) << ", height = " << height);
            const TNode& root = tree.back();
            root.clusteringAt(height, result);
            for (std::size_t i = 0u; i < result.size(); ++i) {
                TSizeVec& ri = result[i];
                std::size_t n = ri.size();
                for (std::size_t j = 0u; j < n; ++j) {
                    ri.insert(ri.end(), H[ri[j]].begin(), H[ri[j]].end());
                }
                ri.erase(ri.begin(), ri.begin() + n);
            }
        } else {
            LOG_ERROR("Failed to cluster " << core::CContainerPrinter::print(heights));
        }
    }

    //! Get the projected data points.
    const TVectorNx1VecVec& projectedData() const { return m_ProjectedData; }

    //! Get the log determinant of the rank full portion of \p m.
    double logDeterminant(const TSvdNxN& svd) const {
        double result = 0.0;
        for (std::size_t i = 0u, rank = static_cast<std::size_t>(svd.rank()); i < rank; ++i) {
            result += std::log(svd.singularValues()[i]);
        }
        return result;
    }

private:
    //! Controls the amount of compression in sampling points
    //! for computing the hierarchical clustering. Larger numbers
    //! equate to more sampled points so less compression.
    double m_Compression;

    //! The projected data points.
    TVectorNx1VecVec m_ProjectedData;
};

//! \brief Adapts clustering implementations for use by the random
//! projection clusterer.
template<typename CLUSTERER>
class CRandomProjectionClustererFacade {};

//! \brief Adapts x-means for use by the random projection clusterer.
template<std::size_t N, typename COST>
class CRandomProjectionClustererFacade<CXMeans<CVectorNx1<double, N>, COST>> {
public:
    using TClusterer = CXMeans<CVectorNx1<double, N>, COST>;
    using TClusterVec = typename TClusterer::TClusterVec;
    using TVectorNx1 = CVectorNx1<double, N>;
    using TVectorNx1Vec = std::vector<TVectorNx1>;

public:
    CRandomProjectionClustererFacade(const TClusterer& xmeans,
                                     std::size_t improveParamsKmeansIterations,
                                     std::size_t improveStructureClusterSeeds,
                                     std::size_t improveStructureKmeansIterations)
        : m_Xmeans(xmeans),
          m_ImproveParamsKmeansIterations(improveParamsKmeansIterations),
          m_ImproveStructureClusterSeeds(improveStructureClusterSeeds),
          m_ImproveStructureKmeansIterations(improveStructureKmeansIterations) {}

    //! Set the points to cluster.
    void setPoints(TVectorNx1Vec& points) { m_Xmeans.setPoints(points); }

    //! Cluster the points.
    void run() { m_Xmeans.run(m_ImproveParamsKmeansIterations, m_ImproveStructureClusterSeeds, m_ImproveStructureKmeansIterations); }

    //! Get the clusters (should only be called after run).
    const TClusterVec& clusters() const { return m_Xmeans.clusters(); }

private:
    //! The x-means implementation.
    TClusterer m_Xmeans;
    //! The number of iterations to use in k-means for a single
    //! round of improve parameters.
    std::size_t m_ImproveParamsKmeansIterations;
    //! The number of random seeds to try when initializing k-means
    //! for a single round of improve structure.
    std::size_t m_ImproveStructureClusterSeeds;
    //! The number of iterations to use in k-means for a single
    //! round of improve structure.
    std::size_t m_ImproveStructureKmeansIterations;
};

//! Makes an x-means adapter for random projection clustering.
template<std::size_t N, typename COST>
CRandomProjectionClustererFacade<CXMeans<CVectorNx1<double, N>, COST>>
forRandomProjectionClusterer(const CXMeans<CVectorNx1<double, N>, COST>& xmeans,
                             std::size_t improveParamsKmeansIterations,
                             std::size_t improveStructureClusterSeeds,
                             std::size_t improveStructureKmeansIterations) {
    return CRandomProjectionClustererFacade<CXMeans<CVectorNx1<double, N>, COST>>(
        xmeans, improveParamsKmeansIterations, improveStructureClusterSeeds, improveStructureKmeansIterations);
}

//! \brief Adapts k-means for use by the random projection clusterer.
template<std::size_t N>
class CRandomProjectionClustererFacade<CKMeansFast<CVectorNx1<double, N>>> {
public:
    using TClusterer = CKMeansFast<CVectorNx1<double, N>>;
    using TClusterVec = typename TClusterer::TClusterVec;
    using TVectorNx1 = CVectorNx1<double, N>;
    using TVectorNx1Vec = std::vector<TVectorNx1>;

public:
    CRandomProjectionClustererFacade(const TClusterer& kmeans, std::size_t k, std::size_t maxIterations)
        : m_Kmeans(kmeans), m_K(k), m_MaxIterations(maxIterations) {}

    //! Set the points to cluster.
    void setPoints(TVectorNx1Vec& points) {
        m_Kmeans.setPoints(points);
        TVectorNx1Vec centres;
        CKMeansPlusPlusInitialization<TVectorNx1, CPRNG::CXorShift1024Mult> seedCentres(m_Rng);
        seedCentres.run(points, m_K, centres);
        m_Kmeans.setCentres(centres);
    }

    //! Cluster the points.
    void run() { m_Kmeans.run(m_MaxIterations); }

    //! Get the clusters (should only be called after run).
    const TClusterVec& clusters() const {
        m_Kmeans.clusters(m_Clusters);
        return m_Clusters;
    }

private:
    //! The random number generator.
    CPRNG::CXorShift1024Mult m_Rng;
    //! The k-means implementation.
    TClusterer m_Kmeans;
    //! The number of clusters to use.
    std::size_t m_K;
    //! The number of iterations to use in k-means.
    std::size_t m_MaxIterations;
    //! The clusters.
    mutable TClusterVec m_Clusters;
};

//! Makes a k-means adapter for random projection clustering.
template<std::size_t N>
CRandomProjectionClustererFacade<CKMeansFast<CVectorNx1<double, N>>>
forRandomProjectionClusterer(const CKMeansFast<CVectorNx1<double, N>>& kmeans, std::size_t k, std::size_t maxIterations) {
    return CRandomProjectionClustererFacade<CKMeansFast<CVectorNx1<double, N>>>(kmeans, k, maxIterations);
}
}
}

#endif // INCLUDED_ml_maths_CRandomProjectionClusterer_h
