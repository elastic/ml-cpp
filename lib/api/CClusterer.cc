/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CClusterer.h>

#include <core/CLogger.h>
#include <core/CStreamUtils.h>
#include <core/CStringUtils.h>

#include <maths/CAnnotatedVector.h>
#include <maths/CClusterEvaluation.h>
#include <maths/CKMeans.h>
#include <maths/CLocalOutlierFactors.h>
#include <maths/CPRNG.h>
#include <maths/CPca.h>
#include <maths/CQuantileSketch.h>
#include <maths/CXMeans.h>

#include <boost/optional.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <cmath>
#include <cstdint>

namespace ml {
namespace api {
namespace {

using TStrVec = std::vector<std::string>;

//! \brief Strips the optionality of a type.
template<typename T>
struct SStripOptional {
    using Type = T;
};

//! \brief Strips the optionality of a type.
template<typename T>
struct SStripOptional<boost::optional<T>> {
    using Type = T;
};

//! Convert the \p source string to the type T.
template<typename T>
bool stringToType(const std::string& source, T& target) {
    return core::CStringUtils::stringToType(source, target);
}

//! Read a general property.
template<typename TYPE>
bool readProperty(const char* typePropertyName,
                  const std::string& propertyName,
                  const std::string& propertyValue,
                  TYPE& property) {
    if (propertyName == typePropertyName) {
        typename SStripOptional<TYPE>::Type property_;
        if (!stringToType(propertyValue, property_)) {
            LOG_ERROR(<< "Bad value for " << propertyName << ": " << propertyValue);
        }
        property = std::move(property_);
        return true;
    }
    return false;
}

//! Read an enumeration property.
template<typename ENUM>
bool readEnumProperty(const char* enumPropertyName,
                      const TStrVec& cases,
                      const std::string& propertyName,
                      const std::string& propertyValue,
                      ENUM& property) {
    if (propertyName == enumPropertyName) {
        auto index = std::find(cases.begin(), cases.end(), propertyValue);
        if (index != cases.end()) {
            property = static_cast<typename SStripOptional<ENUM>::Type>(
                index - cases.begin());
        } else {
            LOG_ERROR(<< "Bad value for enum " << propertyName << ": " << propertyValue);
        }
        return true;
    }
    return false;
}

const std::string CLUSTER_FIELD_VALUE{"cluster_field"};
const std::string CLUSTER_ID{"cluster_id"};
const std::string SILHOUETTE{"silhouette"};
const std::string FEATURE_NAMES{"feature_names"};
const std::string RAW_FEATURE_VECTOR{"raw_vector"};
const std::string TRANSFORMED_FEATURE_VECTOR{"transformed_vector"};
const std::string OUTLIER{"outlier"};
const std::string OUTLIER_FACTOR{"outlier_factor"};

const TStrVec ALGORITHMS{"automatic",
                         "kmeans",
                         "xmeans",
                         "random_projection_kmeans",
                         "random_projection_xmeans",
                         "agglomerative"};
const TStrVec IMPUTATION_METHODS{"random", "nn_plain", "nn_bagged_samples",
                                 "nn_bagged_attributes", "nn_random"};
const TStrVec AGGLOMERATIVE_OBJECTIVES{"single", "complete", "average", "weighted", "ward"};
}

CClusterer::CClusterer(const CParams& params, CClustererOutputWriter& resultWriter)
    : m_Params(params), m_ResultWriter(resultWriter) {
}

void CClusterer::newOutputStream() {
    m_ResultWriter.newOutputStream();
}

bool CClusterer::handleRecord(const TStrStrUMap& fields) {
    auto control = fields.find(CONTROL_FIELD_NAME);
    if (control != fields.end() && !control->second.empty()) {
        return this->handleControlMessage(control->second);
    }

    using TOptionalDouble = boost::optional<double>;

    TOptionalSize cluster;
    TOptionalSize feature;
    TOptionalDouble featureValue;

    for (const auto& field : fields) {
        const std::string& name{field.first};
        std::string value{field.second};
        if (name == CONTROL_FIELD_NAME) {
            // Ignore
        } else if (name == m_Params.clusterField()) {
            cluster.reset(
                m_ToCluster
                    .emplace(m_Dictionary.word(value), m_ToClusterNames.size())
                    .first->second);
            if (*cluster == m_ToClusterNames.size()) {
                m_ToClusterNames.push_back(value);
            }
        } else if (name == m_Params.featureField()) {
            feature.reset(
                m_Dimensions
                    .emplace(m_Dictionary.word(value), m_DimensionNames.size())
                    .first->second);
            if (*feature == m_DimensionNames.size()) {
                m_DimensionNames.push_back(value);
            }
        } else {
            core::CStringUtils::trimWhitespace(value);
            if (!value.empty()) {
                double value_;
                if (!core::CStringUtils::stringToType(value, value_)) {
                    LOG_ERROR(<< "Ignoring bad value for '" << name << "'"
                              << ": '" << value << "'");
                    continue;
                }
                featureValue.reset(value_);
            }
        }
    }

    if (cluster && feature && featureValue) {
        LOG_TRACE(<< "dimensions = " << core::CContainerPrinter::print(m_DimensionNames));
        m_FeatureVectors.resize(std::max(m_FeatureVectors.size(), *cluster + 1));
        m_FeatureVectors[*cluster].emplace_back(*feature, *featureValue);
    } else {
        LOG_ERROR(<< "Ill-formed row missing" << (cluster ? "" : " 'entity label'")
                  << (feature ? "" : " 'feature label'") << (featureValue ? "" : " 'feature value'")
                  << ": " << core::CContainerPrinter::print(fields));
    }

    return true;
}

void CClusterer::finalise() {
    std::ptrdiff_t dimension{static_cast<std::ptrdiff_t>(m_DimensionNames.size())};

    for (auto& vector : m_FeatureVectors) {
        std::sort(vector.begin(), vector.end());
        LOG_TRACE(<< "feature vector = " << core::CContainerPrinter::print(vector));
    }

    if (m_Params.algorithm() == E_Automatic) {
        // TODO determine method based on data characteristics
    }

    this->normalize();

    if (m_Params.treatMissingAsZero()) {
        TSparseVectorVec featureVectors;
        featureVectors.reserve(m_FeatureVectors.size());

        for (auto& vector : m_FeatureVectors) {
            TSparseVector featureVector(dimension);
            for (const auto& feature : vector) {
                featureVector.insertBack(feature.first) = feature.second;
            }

            featureVectors.push_back(std::move(featureVector));
        }

        // TODO sparse clustering
    } else {
        maths::CImputer::EFilterAttributes filter{maths::CImputer::E_NoFiltering};
        maths::CImputer::EMethod method{maths::CImputer::E_NNBaggedAttributes};
        if (!m_Params.imputationMethod()) {
            LOG_DEBUG(<< "No imputation method supplied:"
                         " using nearest neighbours with bags of attributes");
        } else {
            method = *m_Params.imputationMethod();
        }

        maths::CImputer imputer;
        TDenseVectorVec featureVectors;
        imputer.impute(filter, method, dimension, m_FeatureVectors, featureVectors);

        TDoubleVec factors;
        TSizeVec inliers;
        TSizeVec outliers;
        this->removeOutliers(featureVectors, factors, inliers, outliers);

        this->projectOntoPrincipleComponents(inliers, featureVectors);

        TSizeVec clusterIds;
        std::size_t numberClusters{this->cluster(inliers, featureVectors, clusterIds)};
        for (auto i : outliers) {
            clusterIds[i] = numberClusters;
        }

        TDoubleVec silhouettes;
        this->silhouettes(numberClusters, inliers, featureVectors, clusterIds, silhouettes);

        // TODO output explanatory features and/or principle components.
        //TStrVec explaning;
        //this->explaningFeatures(clusters, explanatory);

        this->writeResults(featureVectors, factors, outliers, clusterIds, silhouettes);
    }
}

bool CClusterer::restoreState(core::CDataSearcher& /*restoreSearcher*/,
                              core_t::TTime& /*completeToTime*/) {
    // No-op at present
    return true;
}

bool CClusterer::persistState(core::CDataAdder& /*persister*/) {
    // No-op at present.
    return true;
}

uint64_t CClusterer::numRecordsHandled() const {
    return static_cast<uint64_t>(m_ToCluster.size());
}

COutputHandler& CClusterer::outputHandler() {
    return m_ResultWriter;
}

bool CClusterer::handleControlMessage(const std::string& controlMessage) {
    switch (controlMessage[0]) {
    case 'r':
        this->clear();
        break;
    case 'c':
        this->finalise();
        break;
    default:
        LOG_ERROR(<< "Ignoring unknown control character");
    }
    return true;
}

void CClusterer::clear() {
    m_ToCluster.clear();
    m_Dimensions.clear();
    m_DimensionNames.clear();
    m_FeatureVectors.clear();
}

void CClusterer::normalize() {
    using TQuantileSketchVec = std::vector<maths::CQuantileSketch>;

    if (m_Params.normalize()) {
        LOG_DEBUG(<< "Normalizing " << m_Dimensions.size() << " dimensions");

        TQuantileSketchVec quantiles(
            m_Dimensions.size(),
            maths::CQuantileSketch(maths::CQuantileSketch::E_Linear, 100));
        for (const auto& featureVector : m_FeatureVectors) {
            for (const auto& feature : featureVector) {
                quantiles[feature.first].add(feature.second);
            }
        }

        m_Scales.reserve(m_Dimensions.size());
        for (const auto& quantile : quantiles) {
            double scale{1.0};
            quantile.mad(scale);
            m_Scales.push_back(scale);
        }
        LOG_DEBUG(<< "Scales " << core::CContainerPrinter::print(m_Scales));

        for (auto& featureVector : m_FeatureVectors) {
            for (auto& feature : featureVector) {
                feature.second /= m_Scales[feature.first];
            }
        }
    } else {
        m_Scales.assign(m_DimensionNames.size(), 1.0);
    }
}

void CClusterer::removeOutliers(TDenseVectorVec& vectors,
                                TDoubleVec& factors,
                                TSizeVec& inliers,
                                TSizeVec& outliers) const {
    bool removed{false};
    if (m_Params.computeOutlierFactors()) {
        maths::CLocalOutlierFactors::ensemble(vectors, factors);
        double cutoff{m_Params.outlierCutoff()};
        if (cutoff > 0.0) {
            std::size_t n{static_cast<std::size_t>(
                std::count_if(factors.begin(), factors.end(),
                              [cutoff](double factor) { return factor > cutoff; }))};
            inliers.reserve(vectors.size() - n);
            outliers.reserve(n);
            for (std::size_t i = 0u; i < factors.size(); ++i) {
                (factors[i] > cutoff ? outliers : inliers).push_back(i);
            }
            LOG_DEBUG(<< "Have " << inliers.size() << " inliers"
                      << " and " << outliers.size() << " outliers");
            removed = true;
        }
    }
    if (!removed) {
        inliers.resize(vectors.size());
        std::iota(inliers.begin(), inliers.end(), 0);
    }
}

void CClusterer::projectOntoPrincipleComponents(const TSizeVec& inliers,
                                                TDenseVectorVec& vectors) const {
    if (m_Params.principleComponents() && !vectors.empty()) {
        LOG_DEBUG(<< "Projecting data onto " << *m_Params.principleComponents()
                  << " principle components");
        maths::CPca::projectOntoPrincipleComponents(*m_Params.principleComponents(),
                                                    inliers, vectors);
    }
}

std::size_t CClusterer::cluster(const TSizeVec& inliers,
                                TDenseVectorVec& vectors_,
                                TSizeVec& clusterIds) const {
    LOG_DEBUG(<< "Clustering...");

    clusterIds.resize(m_ToCluster.size());

    TAnnotatedDenseVectorVec vectors;
    for (auto i : inliers) {
        vectors.emplace_back(std::move(vectors_[i]), i);
    }

    std::size_t result{0};
    switch (m_Params.algorithm()) {
    case E_Automatic:
        break;
    case E_KMeans:
        if (m_Params.numberOfClusters()) {
            std::size_t k{*m_Params.numberOfClusters()};
            LOG_DEBUG(<< k << "-means");
            result = this->kmeans(k, vectors, clusterIds);
        } else {
            LOG_ERROR(<< "Selected k-means but haven't set k");
        }
        break;
    case E_XMeans:
        if (m_Params.maximumNumberOfClusters()) {
            std::size_t max{*m_Params.maximumNumberOfClusters()};
            LOG_TRACE(<< "x-means with max clusters " << max);
            result = this->xmeans(max, vectors, clusterIds);
        } else {
            LOG_ERROR(<< "Selected x-means but haven't selected maximum number of clusters");
        }
        break;
    case E_RandomProjectionKMeans:
    case E_RandomProjectionXMeans:
    case E_Agglomerative:
        break;
    }

    for (auto& vector : vectors) {
        vectors_[vector.annotation()] = std::move(static_cast<TDenseVector&>(vector));
    }

    return result;
}

std::size_t CClusterer::kmeans(std::size_t k,
                               TAnnotatedDenseVectorVec& vectors,
                               TSizeVec& clusterIds) const {
    using TMinAccumulator = maths::CBasicStatistics::SMin<double>::TAccumulator;
    using TKMeansInitializer =
        maths::CKMeansPlusPlusInitialization<TAnnotatedDenseVector, maths::CPRNG::CXorShift1024Mult>;
    using TKMeans = maths::CKMeans<TAnnotatedDenseVector>;

    if (k < vectors.size()) {
        maths::CPRNG::CXorShift1024Mult rng;
        TKMeansInitializer initializer(rng);
        TMinAccumulator minimumDispersion;
        for (std::size_t run = 0u; run < 10; ++run) {
            TAnnotatedDenseVectorVec seed;
            initializer.run(vectors, k, seed);

            TKMeans kmeans;
            kmeans.reserve(vectors.size());
            kmeans.setPoints(vectors);
            kmeans.setCentres(seed);
            kmeans.run(20);

            const TAnnotatedDenseVectorVec& centres{kmeans.centres()};
            TAnnotatedDenseVectorVecVec clusters;
            kmeans.clusters(clusters);

            double dispersion{TKMeans::dispersion(centres, clusters)};
            LOG_TRACE(<< "dispersion = " << dispersion);

            if (minimumDispersion.add(dispersion)) {
                for (std::size_t i = 0u; i < clusters.size(); ++i) {
                    for (const auto& vector : clusters[i]) {
                        clusterIds[vector.annotation()] = i;
                    }
                }
            }
        }
        return k;
    }

    for (std::size_t i = 0u; i < vectors.size(); ++i) {
        clusterIds[vectors[i].annotation()] = i;
    }
    return vectors.size();
}

std::size_t CClusterer::xmeans(std::size_t max,
                               TAnnotatedDenseVectorVec& vectors,
                               TSizeVec& clusterIds) const {
    using TXMeans =
        maths::CXMeans<TAnnotatedDenseVector, maths::CGaussianInfoCriterion<TAnnotatedDenseVector, maths::E_BIC>>;

    TXMeans xmeans(max);
    xmeans.setPoints(vectors);
    xmeans.run(3, 3, 3);
    for (std::size_t i = 0u; i < xmeans.clusters().size(); ++i) {
        for (const auto& vector : xmeans.clusters()[i].points()) {
            clusterIds[vector.annotation()] = i;
        }
    }
    return xmeans.clusters().size();
}

void CClusterer::silhouettes(std::size_t numberClusters,
                             const TSizeVec& inliers,
                             TDenseVectorVec& vectors,
                             const TSizeVec& clusterIds,
                             TDoubleVec& silhouettes) const {
    TAnnotatedDenseVectorVecVec clusters;
    clusters.assign(numberClusters, TAnnotatedDenseVectorVec());
    for (auto i : inliers) {
        clusters[clusterIds[i]].emplace_back(std::move(vectors[i]), i);
    }

    TDoubleVecVec silhouettes_, errors;
    maths::CClusterEvaluation::silhouetteApprox(clusters, silhouettes_, errors);

    silhouettes.resize(vectors.size(), -1.0);
    for (std::size_t i = 0u; i < clusters.size(); ++i) {
        for (std::size_t j = 0u; j < clusters[i].size(); ++j) {
            std::size_t index{clusters[i][j].annotation()};
            silhouettes[index] = silhouettes_[i][j];
            vectors[index] = std::move(static_cast<TDenseVector&>(clusters[i][j]));
        }
    }
}

void CClusterer::explaningFeatures(const TDenseVectorVecVec& clusters, TStrVec& result) const {
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;

    result.clear();

    std::size_t dimension{m_DimensionNames.size()};
    if (dimension <= 4) {
        result = m_DimensionNames;
        return;
    }

    TMeanAccumulatorVecVec silhouettes(dimension * (dimension / 2),
                                       TMeanAccumulatorVec(clusters.size()));

    TDenseVectorVecVec projections(clusters.size());
    TDoubleVecVec clusterSilhouettes;
    TDoubleVecVec errors;
    for (std::size_t i = 0u, s = 0u; i < dimension; ++i) {
        for (std::size_t j = 0u; j < i; ++j, ++s) {
            for (std::size_t k = 0u; k < clusters.size(); ++k) {
                projections[k].resize(clusters[k].size(), TDenseVector(2));
                for (std::size_t p = 0u; p < clusters[k].size(); ++p) {
                    projections[k][p][0] = clusters[k][p][i];
                    projections[k][p][1] = clusters[k][p][j];
                }
                maths::CClusterEvaluation::silhouetteApprox(
                    projections, clusterSilhouettes, errors);
                silhouettes[s][k].add(clusterSilhouettes);
            }
        }
    }

    // We choose the coordinate set C of cardinality 4, so we can
    // easily plot all combinations, which greedily maximizes the
    // clusterings average silhouette over the set of all coordinate
    // pairs. Specifically, we greedily maximize
    //
    //   sum_{cluster}{max_{(i,j) in C}{sum{silhouette({P_{i,j}(x) | x in cluster}}}}
    //
    // Note the following makes use of the fact that there's a bijection
    // from the pairs (i, j) -> k being the index into the silhouette
    // statistics. In particular, k = (i+1) x i / 2 + j for j < i whence
    // i = floor((sqrt(1 + 8 x k) - 1) / 2) and j = k - (i+1) x i / 2.

    auto less = [](const TMeanAccumulatorVec& lhs, const TMeanAccumulatorVec& rhs) {
        return maths::CBasicStatistics::mean(
                   std::accumulate(lhs.begin(), lhs.end(), TMeanAccumulator())) <=
               maths::CBasicStatistics::mean(
                   std::accumulate(rhs.begin(), rhs.end(), TMeanAccumulator()));
    };

    std::size_t k{static_cast<std::size_t>(
        std::max_element(silhouettes.begin(), silhouettes.end(), less) -
        silhouettes.begin())};
    std::size_t i{static_cast<std::size_t>(
        std::sqrt((8.0 * static_cast<double>(k) + 1.0) - 1.0) / 2.0)};
    std::size_t j{k - i * (i + 1) / 2};
    TSizeVec C{i, j};
    TMeanAccumulatorVec max(silhouettes[k]);
    TMeanAccumulatorVec trial;
    while (C.size() < 4) {
        C.push_back(dimension);
        for (std::size_t u = 0u; u < dimension; ++u) {
            if (std::find(C.begin(), C.end(), u) == C.end()) {
                trial = max;
                for (auto v : C) {
                    k = std::max(u, v) * (std::max(u, v) + 1) / 2 + std::min(u, v);
                    for (std::size_t cluster = 0u; cluster < trial.size(); ++cluster) {
                        trial[cluster] =
                            maths::CBasicStatistics::mean(trial[cluster]) <
                                    maths::CBasicStatistics::mean(silhouettes[k][cluster])
                                ? silhouettes[k][cluster]
                                : trial[cluster];
                    }
                }
                if (less(max, trial)) {
                    C.back() = u;
                    max = std::move(trial);
                }
            }
        }
    }

    result.reserve(C.size());
    for (auto coordinate : C) {
        result.push_back(m_DimensionNames[coordinate]);
    }
}

void CClusterer::writeResults(const TDenseVectorVec& featureVectors,
                              const TDoubleVec& factors,
                              const TSizeVec& outliers,
                              const TSizeVec& clusterIds,
                              TDoubleVec& silhouettes) const {
    using TStrDoubleUMap = boost::unordered_map<std::string, double>;

    m_ResultWriter.startResult();
    m_ResultWriter.addMember(FEATURE_NAMES, m_DimensionNames);
    m_ResultWriter.writeResult();

    TStrDoubleUMap raw;
    TStrDoubleUMap transformed;
    for (std::size_t i = 0u; i < m_ToClusterNames.size(); ++i) {
        raw.clear();
        for (const auto& feature : m_FeatureVectors[i]) {
            std::ptrdiff_t j{feature.first};
            raw[m_DimensionNames[j]] = m_Scales[j] * feature.second;
        }

        transformed.clear();
        for (std::ptrdiff_t j = 0; j < featureVectors[i].size(); ++j) {
            transformed[m_DimensionNames[j]] = featureVectors[i](j);
        }

        m_ResultWriter.startResult();
        m_ResultWriter.addMember(CLUSTER_FIELD_VALUE, m_ToClusterNames[i]);
        m_ResultWriter.addMember(CLUSTER_ID, clusterIds[i]);
        m_ResultWriter.addMember(SILHOUETTE, silhouettes[i]);
        m_ResultWriter.addMember(RAW_FEATURE_VECTOR, raw);
        m_ResultWriter.addMember(TRANSFORMED_FEATURE_VECTOR, transformed);
        m_ResultWriter.addMember(
            OUTLIER, std::binary_search(outliers.begin(), outliers.end(), i));
        m_ResultWriter.addMember(OUTLIER_FACTOR, factors[i]);
        m_ResultWriter.writeResult();
    }
}

CClusterer::CParams::CParams()
    : m_Algorithm(E_Automatic), m_Normalize(false), m_TreatMissingAsZero(false),
      m_ComputeOutlierFactors(false), m_OutlierCutoff(1.0), m_Bootstrap(true) {
}

bool CClusterer::CParams::init(const std::string& configFile) {
    LOG_DEBUG(<< "Reading config file " << configFile);

    boost::property_tree::ptree properties;
    try {
        std::ifstream strm(configFile.c_str());
        if (!strm.is_open()) {
            LOG_ERROR(<< "Error opening config file " << configFile);
            return false;
        }
        core::CStreamUtils::skipUtf8Bom(strm);

        boost::property_tree::ini_parser::read_ini(strm, properties);
    } catch (boost::property_tree::ptree_error& e) {
        LOG_ERROR(<< "Error reading config file " << configFile << " : " << e.what());
        return false;
    }

    for (const auto& property : properties) {
        const std::string& propertyName = property.first;
        std::string propertyValue = property.second.data();
        core::CStringUtils::trimWhitespace(propertyValue);
        core::CStringUtils::toLower(propertyValue);
        if (readEnumProperty("algorithm", ALGORITHMS, propertyName, propertyValue, m_Algorithm)) {
            continue;
        }
        if (readEnumProperty("imputation_method", IMPUTATION_METHODS,
                             propertyName, propertyValue, m_ImputationMethod)) {
            continue;
        }
        if (readEnumProperty("agglomerative_objective", AGGLOMERATIVE_OBJECTIVES,
                             propertyName, propertyValue, m_AgglomerativeObjective)) {
            continue;
        }
        if (readProperty("normalize", propertyName, propertyValue, m_Normalize)) {
            continue;
        }
        if (readProperty("treat_missing_as_zero", propertyName, propertyValue,
                         m_TreatMissingAsZero)) {
            continue;
        }
        if (readProperty("compute_outlier_factors", propertyName, propertyValue,
                         m_ComputeOutlierFactors)) {
            continue;
        }
        if (readProperty("outlier_cutoff", propertyName, propertyValue, m_OutlierCutoff)) {
            continue;
        }
        if (readProperty("bootstrap", propertyName, propertyValue, m_Bootstrap)) {
            continue;
        }
        if (readProperty("number_of_clusters", propertyName, propertyValue, m_NumberOfClusters)) {
            continue;
        }
        if (readProperty("maximum_number_of_clusters", propertyName,
                         propertyValue, m_MaximumNumberOfClusters)) {
            continue;
        }
        if (readProperty("principle_components", propertyName, propertyValue,
                         m_PrincipleComponents)) {
            continue;
        }
    }

    return true;
}

CClusterer::CParams& CClusterer::CParams::clusterField(const std::string& clusterField) {
    m_ClusterField = clusterField;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::featureField(const std::string& featureField) {
    m_FeatureField = featureField;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::algorithm(EAlgorithm algorithm) {
    m_Algorithm = algorithm;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::normalize(bool normalize) {
    m_Normalize = normalize;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::treatMissingAsZero(bool treatMissingAsZero) {
    m_TreatMissingAsZero = treatMissingAsZero;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::imputationMethod(maths::CImputer::EMethod imputationMethod) {
    m_ImputationMethod = imputationMethod;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::computeOutlierFactors(bool compute) {
    m_ComputeOutlierFactors = compute;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::outlierCutoff(double cutoff) {
    m_OutlierCutoff = cutoff;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::bootstrap(bool bootstrap) {
    m_Bootstrap = bootstrap;
    return *this;
}

CClusterer::CParams& CClusterer::CParams::numberOfClusters(std::size_t numberOfClusters) {
    m_NumberOfClusters.reset(numberOfClusters);
    return *this;
}

CClusterer::CParams& CClusterer::CParams::maximumNumberOfClusters(std::size_t maximumNumberOfClusters) {
    m_MaximumNumberOfClusters.reset(maximumNumberOfClusters);
    return *this;
}

CClusterer::CParams& CClusterer::CParams::principleComponents(std::size_t principleComponents) {
    m_PrincipleComponents.reset(principleComponents);
    return *this;
}

CClusterer::CParams& CClusterer::CParams::agglomerativeObjective(
    maths::CAgglomerativeClusterer::EObjective agglomerativeObjective) {
    m_AgglomerativeObjective.reset(agglomerativeObjective);
    return *this;
}

const std::string& CClusterer::CParams::clusterField() const {
    return m_ClusterField;
}

const std::string& CClusterer::CParams::featureField() const {
    return m_FeatureField;
}

CClusterer::EAlgorithm CClusterer::CParams::algorithm() const {
    return m_Algorithm;
}

bool CClusterer::CParams::normalize() const {
    return m_Normalize;
}

bool CClusterer::CParams::treatMissingAsZero() const {
    return m_TreatMissingAsZero;
}

CClusterer::TOptionalImputationMethod CClusterer::CParams::imputationMethod() const {
    return m_ImputationMethod;
}

bool CClusterer::CParams::computeOutlierFactors() const {
    return m_ComputeOutlierFactors;
}

double CClusterer::CParams::outlierCutoff() const {
    return m_OutlierCutoff;
}

bool CClusterer::CParams::bootstrap() const {
    return m_Bootstrap;
}

CClusterer::TOptionalSize CClusterer::CParams::numberOfClusters() const {
    return m_NumberOfClusters;
}

CClusterer::TOptionalSize CClusterer::CParams::maximumNumberOfClusters() const {
    return m_MaximumNumberOfClusters;
}

CClusterer::TOptionalSize CClusterer::CParams::principleComponents() const {
    return m_PrincipleComponents;
}

CClusterer::TOptionalObjective CClusterer::CParams::agglomerativeObjective() const {
    return m_AgglomerativeObjective;
}
}
}
