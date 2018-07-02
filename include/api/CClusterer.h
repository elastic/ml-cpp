/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CClusterer_h
#define INCLUDED_ml_api_CClusterer_h

#include <core/CCompressedDictionary.h>

#include <maths/CAgglomerativeClusterer.h>
#include <maths/CImputer.h>
#include <maths/CLinearAlgebraEigen.h>

#include <api/CClustererOutputWriter.h>
#include <api/CDataProcessor.h>
#include <api/ImportExport.h>

#include <boost/optional.hpp>

#include <cstddef>
#include <vector>

namespace ml {
namespace api {

//! \brief Wraps up various clustering utilities.
class API_EXPORT CClusterer : public CDataProcessor {
public:
    using TOptionalSize = boost::optional<std::size_t>;
    using TOptionalImputationMethod = boost::optional<maths::CImputer::EMethod>;
    using TOptionalObjective = boost::optional<maths::CAgglomerativeClusterer::EObjective>;

    //! The possible algorithms.
    enum EAlgorithm {
        E_Automatic, //!< Select the algorithm based on data characteristics.
        E_KMeans,    //!< k-means clustering.
        E_XMeans,    //!< x-means clustering.
        E_RandomProjectionKMeans, //!< Random projection clustering with k-means.
        E_RandomProjectionXMeans, //!< Random projection clustering with x-means.
        E_Agglomerative           //!< Hierarchical agglomerative clustering.
    };

    //! \brief The clustering parameters.
    class API_EXPORT CParams {
    public:
        CParams();

        bool init(const std::string& configFile);
        CParams& clusterField(const std::string& clusterField);
        CParams& featureField(const std::string& featureField);
        CParams& algorithm(EAlgorithm algorithm);
        CParams& normalize(bool normalize);
        CParams& treatMissingAsZero(bool treatMissingAsZero);
        CParams& imputationMethod(maths::CImputer::EMethod imputationMethod);
        CParams& computeOutlierFactors(bool compute);
        CParams& outlierCutoff(double cutoff);
        CParams& bootstrap(bool bootstrap);
        CParams& numberOfClusters(std::size_t numberOfClusters);
        CParams& maximumNumberOfClusters(std::size_t maximumNumberOfClusters);
        CParams& principleComponents(std::size_t principleComponents);
        CParams& agglomerativeObjective(maths::CAgglomerativeClusterer::EObjective agglomerativeObjective);

        //! Get the field to cluster.
        const std::string& clusterField() const;
        //! Get the field which identifies features.
        const std::string& featureField() const;
        //! Get the clustering algorithm to use.
        EAlgorithm algorithm() const;
        //! Check if we should normalize each coordinate separately.
        bool normalize() const;
        //! Check if we should treat missing values as zero.
        bool treatMissingAsZero() const;
        //! Get the imputation method if it has been set.
        TOptionalImputationMethod imputationMethod() const;
        //! Check whether or not to compute outlier factors.
        bool computeOutlierFactors() const;
        //! Get cutoff threshold for removing outliers.
        double outlierCutoff() const;
        //! Check whether or not to bootstrap the clustering.
        bool bootstrap() const;
        //! Get the number of clusters to use if the algorithm allows one
        //! to choose.
        TOptionalSize numberOfClusters() const;
        //! Get the maximum number of clusters to use if we're searching
        //! for the appropriate number.
        TOptionalSize maximumNumberOfClusters() const;
        //! Project the data onto this number of principle components.
        TOptionalSize principleComponents() const;
        //! Get the objective to use in agglomerative clustering.
        TOptionalObjective agglomerativeObjective() const;

    private:
        //! The field to cluster.
        std::string m_ClusterField;
        //! The field which identifies features.
        std::string m_FeatureField;
        //! The algorithm with which to cluster.
        EAlgorithm m_Algorithm;
        //! Normalize the individual coordinates separately.
        bool m_Normalize;
        //! Treat missing values as zero. This is appropriate for count
        //! features for example.
        bool m_TreatMissingAsZero;
        //! The method by which to impute missing data.
        TOptionalImputationMethod m_ImputationMethod;
        //! Compute the feature vector outlier factors.
        bool m_ComputeOutlierFactors;
        //! The cutoff factor to treat a feature vector as an outlier.
        double m_OutlierCutoff;
        //! Bootstrap the clustering.
        bool m_Bootstrap;
        //! The number of clusters to produce if specified. This is ignored
        //! by algorithms which compute the number of clusters.
        TOptionalSize m_NumberOfClusters;
        //! The maximum number of clusters for algorithms which search for
        //! for the appropriate number of clusters.
        TOptionalSize m_MaximumNumberOfClusters;
        //! Project the data onto this number of principle components.
        TOptionalSize m_PrincipleComponents;
        //! The objective to use in agglomerative clustering.
        TOptionalObjective m_AgglomerativeObjective;
    };

public:
    CClusterer(const CParams& params, CClustererOutputWriter& resultWriter);

    //! Change the output stream to which to write results.
    virtual void newOutputStream();

    //! Receive a single labeled entity and its feature vector.
    virtual bool handleRecord(const TStrStrUMap& fields);

    //! Do the clustering.
    virtual void finalise();

    //! Restore the previous clustering.
    virtual bool restoreState(core::CDataSearcher& restoreSearcher,
                              core_t::TTime& completeToTime);

    //! Persist current clustering.
    virtual bool persistState(core::CDataAdder& persister);

    //! How many records did we handle?
    virtual uint64_t numRecordsHandled() const;

    //! Access the output handler.
    virtual COutputHandler& outputHandler();

private:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TDictionary = core::CCompressedDictionary<2>;
    using TWordSizeUMap = TDictionary::TWordTUMap<std::size_t>;
    using TIntDoublePr = std::pair<std::ptrdiff_t, double>;
    using TIntDoublePrVec = std::vector<TIntDoublePr>;
    using TIntDoublePrVecVec = std::vector<TIntDoublePrVec>;
    using TDenseVector = maths::CDenseVector<double>;
    using TDenseVectorVec = std::vector<TDenseVector>;
    using TDenseVectorVecVec = std::vector<TDenseVectorVec>;
    using TSparseVector = maths::CSparseVector<double>;
    using TSparseVectorVec = std::vector<TSparseVector>;
    using TAnnotatedDenseVector = maths::CAnnotatedVector<TDenseVector, std::size_t>;
    using TAnnotatedDenseVectorVec = std::vector<TAnnotatedDenseVector>;
    using TAnnotatedDenseVectorVecVec = std::vector<TAnnotatedDenseVectorVec>;

private:
    //! Handle a control message. The first character of the control
    //! message indicates its type. Currently defined types are:
    //! ' ' => Clear the state currently being maintained.
    //! 'c' => Cluster the data sent so far.
    bool handleControlMessage(const std::string& controlMessage);

    //! Clear all state.
    void clear();

    //! Normalize each dimension separately.
    void normalize();

    //! Remove outliers, if we've been asked to, and compute vector
    //! outlier scores.
    void removeOutliers(TDenseVectorVec& vectors,
                        TDoubleVec& factors,
                        TSizeVec& inliers,
                        TSizeVec& outliers) const;

    //! If we've been asked to, project the data on to the specified
    //! number of principle components.
    void projectOntoPrincipleComponents(const TSizeVec& inliers, TDenseVectorVec& vectors) const;

    //! Perform clustering and fill in \p clusterIds with the mapping
    //! from objects to cluster to cluster identifier.
    std::size_t cluster(const TSizeVec& inliers, TDenseVectorVec& vectors, TSizeVec& clusterIds) const;

    //! Cluster \p vectors using k-means with \p k centroids.
    std::size_t kmeans(std::size_t k, TAnnotatedDenseVectorVec& vectors, TSizeVec& clusterIds) const;

    //! Cluster \p vectors using x-means with maximum centroids \p max.
    std::size_t xmeans(std::size_t max, TAnnotatedDenseVectorVec& vectors, TSizeVec& clusterIds) const;

    //! Compute the vector's silhouette statistics.
    void silhouettes(std::size_t numberClusters,
                     const TSizeVec& inliers,
                     TDenseVectorVec& vectors,
                     const TSizeVec& clusterIds,
                     TDoubleVec& silhouettes) const;

    //! Find the most relevant coordinates as the coordinate pairs which
    //! collectively maximize the mean silhouette score.
    void explaningFeatures(const TDenseVectorVecVec& clusters, TStrVec& result) const;

    //! Write out the results of clustering.
    void writeResults(const TDenseVectorVec& featureVectors,
                      const TDoubleVec& factors,
                      const TSizeVec& outliers,
                      const TSizeVec& clusterIds,
                      TDoubleVec& silhouettes) const;

private:
    //! The cluster parameters.
    CParams m_Params;

    //! The clustering result writer.
    CClustererOutputWriter& m_ResultWriter;

    //! A compressed dictionary.
    TDictionary m_Dictionary;

    //! The index of each object to cluster in the m_ToClusterNames vector.
    TWordSizeUMap m_ToCluster;

    //! The names of the objects to cluster.
    TStrVec m_ToClusterNames;

    //! The index of each dimension name in the m_DimensionNames vector.
    TWordSizeUMap m_Dimensions;

    //! Holds the name of each dimension.
    TStrVec m_DimensionNames;

    //! Holds the feature vectors.
    TIntDoublePrVecVec m_FeatureVectors;

    //! Holds scales we've applied to the each dimension.
    TDoubleVec m_Scales;
};
}
}

#endif // INCLUDED_ml_api_CClusterer_h
