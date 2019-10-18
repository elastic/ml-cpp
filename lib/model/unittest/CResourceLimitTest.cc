/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CRegex.h>

#include <maths/CIntegerTools.h>

#include <model/CAnomalyDetector.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CEventRateModel.h>
#include <model/CEventRateModelFactory.h>
#include <model/CEventRatePopulationModel.h>
#include <model/CEventRatePopulationModelFactory.h>
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsAggregator.h>
#include <model/CHierarchicalResultsPopulator.h>
#include <model/CHierarchicalResultsProbabilityFinalizer.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CLimits.h>
#include <model/CMetricModel.h>
#include <model/CMetricModelFactory.h>
#include <model/CResourceMonitor.h>

#include <boost/test/unit_test.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CResourceLimitTest)

using namespace ml;
using namespace model;

using TStrVec = std::vector<std::string>;

class CResultWriter : public ml::model::CHierarchicalResultsVisitor {
public:
    using TResultsTp =
        boost::tuple<core_t::TTime, double /* probability */, std::string /* byFieldName*/, std::string /* overFieldName */, std::string /* partitionFieldName */>;
    using TResultsVec = std::vector<TResultsTp>;

public:
    CResultWriter(const CAnomalyDetectorModelConfig& modelConfig, const CLimits& limits)
        : m_ModelConfig(modelConfig), m_Limits(limits) {}

    void operator()(CAnomalyDetector& detector, core_t::TTime start, core_t::TTime end) {
        CHierarchicalResults results;
        detector.buildResults(start, end, results);
        results.buildHierarchy();
        CHierarchicalResultsAggregator aggregator(m_ModelConfig);
        results.bottomUpBreadthFirst(aggregator);
        model::CHierarchicalResultsProbabilityFinalizer finalizer;
        results.bottomUpBreadthFirst(finalizer);
        model::CHierarchicalResultsPopulator populator(m_Limits);
        results.bottomUpBreadthFirst(populator);
        results.bottomUpBreadthFirst(*this);
    }

    virtual void visit(const ml::model::CHierarchicalResults& results,
                       const ml::model::CHierarchicalResults::TNode& node,
                       bool pivot) {
        if (pivot) {
            return;
        }
        if (!this->shouldWriteResult(m_Limits, results, node, pivot)) {
            return;
        }
        if (this->isSimpleCount(node)) {
            return;
        }
        if (!this->isLeaf(node)) {
            return;
        }

        LOG_DEBUG(<< "Got anomaly @ " << node.s_BucketStartTime << ": "
                  << node.probability());

        ml::model::SAnnotatedProbability::TAttributeProbability1Vec& attributes =
            node.s_AnnotatedProbability.s_AttributeProbabilities;

        m_Results.push_back(TResultsTp(
            node.s_BucketStartTime, node.probability(),
            (attributes.empty() ? "" : *attributes[0].s_Attribute),
            *node.s_Spec.s_PersonFieldValue, *node.s_Spec.s_PartitionFieldValue));
    }

    bool operator()(ml::core_t::TTime time,
                    const ml::model::CHierarchicalResults::TNode& node,
                    bool isBucketInfluencer) {
        LOG_DEBUG(<< (isBucketInfluencer ? "BucketInfluencer" : "Influencer ")
                  << node.s_Spec.print() << " initial score "
                  << node.probability() << ", time:  " << time);

        return true;
    }

    const TResultsVec& results() const { return m_Results; }

private:
    const CAnomalyDetectorModelConfig& m_ModelConfig;
    const CLimits& m_Limits;
    TResultsVec m_Results;
};


BOOST_AUTO_TEST_CASE(testLimitBy) {
    // Check that we can get some results from a test data set, then
    // turn on resource limiting and still get the same results

    static const core_t::TTime BUCKET_LENGTH(3600);
    static const core_t::TTime FIRST_TIME(
        maths::CIntegerTools::ceil(core_t::TTime(1407428000), BUCKET_LENGTH));
    ::CResultWriter::TResultsVec results;

    {
        CAnomalyDetectorModelConfig modelConfig =
            CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);
        modelConfig.useMultibucketFeatures(false);
        CLimits limits;
        CSearchKey key(1, // identifier
                       function_t::E_IndividualMetric, false,
                       model_t::E_XF_None, "value", "colour");
        CAnomalyDetector detector(1, // identifier
                                  limits, modelConfig, "", FIRST_TIME,
                                  modelConfig.factory(key));
        ::CResultWriter writer(modelConfig, limits);

        importCsvDataWithLimiter(FIRST_TIME, BUCKET_LENGTH, writer,
                                 "testfiles/resource_limits_8_series.csv",
                                 detector, std::numeric_limits<std::size_t>::max(),
                                 limits.resourceMonitor());

        results = writer.results();

        // expect there to be 2 anomalies
        BOOST_CHECK_EQUAL(std::size_t(2), results.size());
        BOOST_CHECK_EQUAL(core_t::TTime(1407571200), results[0].get<0>());
        BOOST_CHECK_EQUAL(core_t::TTime(1407715200), results[1].get<0>());
        BOOST_CHECK_EQUAL(std::size_t(8), detector.numberActivePeople());
    }
    {
        // This time, repeat the test but set a resource limit to prevent
        // any models from being created.
        CAnomalyDetectorModelConfig modelConfig =
            CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);
        CLimits limits;
        CSearchKey key(1, // identifier
                       function_t::E_IndividualMetric, false,
                       model_t::E_XF_None, "value", "colour");
        CAnomalyDetector detector(1, // identifier
                                  limits, modelConfig, "", FIRST_TIME,
                                  modelConfig.factory(key));
        ::CResultWriter writer(modelConfig, limits);

        importCsvDataWithLimiter(FIRST_TIME, BUCKET_LENGTH, writer,
                                 "testfiles/resource_limits_8_series.csv",
                                 detector, 1, limits.resourceMonitor());

        const ::CResultWriter::TResultsVec& secondResults = writer.results();

        BOOST_CHECK_EQUAL(std::size_t(0), secondResults.size());
    }
}

BOOST_AUTO_TEST_CASE(testLimitByOver) {
    // Check that we can get some results from a test data set, then
    // turn on resource limiting and still get the results from
    // non-limited data, but not results from limited data

    static const core_t::TTime BUCKET_LENGTH(3600);
    static const core_t::TTime FIRST_TIME(
        maths::CIntegerTools::ceil(core_t::TTime(1407441600), BUCKET_LENGTH));
    ::CResultWriter::TResultsVec results;

    {
        CAnomalyDetectorModelConfig modelConfig =
            CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);
        CLimits limits;
        CSearchKey key(1, // identifier
                       function_t::E_PopulationMetric, false,
                       model_t::E_XF_None, "value", "colour", "species");
        CAnomalyDetector detector(1, // identifier
                                  limits, modelConfig, "", FIRST_TIME,
                                  modelConfig.factory(key));
        ::CResultWriter writer(modelConfig, limits);

        importCsvDataWithLimiter(FIRST_TIME, BUCKET_LENGTH, writer,
                                 "testfiles/resource_limits_8_2over.csv", detector,
                                 std::numeric_limits<std::size_t>::max(),
                                 limits.resourceMonitor());

        results = writer.results();

        // check we have the expected 4 anomalies
        BOOST_CHECK_EQUAL(std::size_t(4), results.size());
        BOOST_CHECK_EQUAL(std::size_t(2), detector.numberActivePeople());
        BOOST_CHECK_EQUAL(std::size_t(3), detector.numberActiveAttributes());
    }

    // Now limit after 1 sample, so only expect no results
    CAnomalyDetectorModelConfig modelConfig =
        CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);
    CLimits limits;
    CSearchKey key(1, // identifier
                   function_t::E_PopulationMetric, false, model_t::E_XF_None,
                   "value", "colour", "species");
    CAnomalyDetector detector(1, // identifier
                              limits, modelConfig, "", FIRST_TIME,
                              modelConfig.factory(key));
    ::CResultWriter writer(modelConfig, limits);

    importCsvDataWithLimiter(FIRST_TIME, BUCKET_LENGTH, writer,
                             "testfiles/resource_limits_8_2over.csv", detector,
                             1, limits.resourceMonitor());

    const ::CResultWriter::TResultsVec& secondResults = writer.results();

    // should only have red flowers as results now
    BOOST_CHECK_EQUAL(std::size_t(0), secondResults.size());
}

namespace {

class CMockModelInterface {
public:
    CMockModelInterface() = default;
    virtual ~CMockModelInterface() = default;

    virtual void createNewModels(std::size_t n, std::size_t m) = 0;

    virtual void test(core_t::TTime time) = 0;

    virtual std::size_t getNewPeople() const = 0;

    virtual std::size_t getNewAttributes() const = 0;
};

//! A test wrapper around a real model that tracks calls to createNewModels
//! and simulates taking lots of memory
class CMockEventRateModel : public CEventRateModel, public CMockModelInterface {
public:
    CMockEventRateModel(const SModelParams& params,
                        const TDataGathererPtr& dataGatherer,
                        const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                        const maths::CMultinomialConjugate& personProbabilityPrior,
                        const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
                        CResourceMonitor& resourceMonitor)
        : CEventRateModel(params,
                          dataGatherer,
                          newFeatureModels,
                          TFeatureMultivariatePriorSPtrPrVec(),
                          TFeatureCorrelationsPtrPrVec(),
                          personProbabilityPrior,
                          influenceCalculators,
                          std::make_shared<CInterimBucketCorrector>(params.s_BucketLength)),
          m_ResourceMonitor(resourceMonitor), m_NewPeople(0), m_NewAttributes(0) {}

    virtual void updateRecycledModels() {
        // Do nothing
    }

    virtual void createNewModels(std::size_t n, std::size_t m) {
        m_NewPeople += n;
        m_NewAttributes += m;
        this->CEventRateModel::createNewModels(n, m);
    }

    void test(core_t::TTime time) {
        m_ResourceMonitor.clearExtraMemory();
        this->createUpdateNewModels(time, m_ResourceMonitor);
    }

    std::size_t getNewPeople() const { return m_NewPeople; }

    std::size_t getNewAttributes() const { return m_NewAttributes; }

private:
    CResourceMonitor& m_ResourceMonitor;
    std::size_t m_NewPeople;
    std::size_t m_NewAttributes;
};

//! A test wrapper around a real model that tracks calls to createNewModels
//! and simulates taking lots of memory
class CMockMetricModel : public CMetricModel, public CMockModelInterface {
public:
    CMockMetricModel(const SModelParams& params,
                     const TDataGathererPtr& dataGatherer,
                     const TFeatureMathsModelSPtrPrVec& newFeatureModels,
                     const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators,
                     CResourceMonitor& resourceMonitor)
        : CMetricModel(params,
                       dataGatherer,
                       newFeatureModels,
                       TFeatureMultivariatePriorSPtrPrVec(),
                       TFeatureCorrelationsPtrPrVec(),
                       influenceCalculators,
                       std::make_shared<CInterimBucketCorrector>(params.s_BucketLength)),
          m_ResourceMonitor(resourceMonitor), m_NewPeople(0), m_NewAttributes(0) {}

    virtual void updateRecycledModels() {
        // Do nothing
    }

    virtual void createNewModels(std::size_t n, std::size_t m) {
        m_NewPeople += n;
        m_NewAttributes += m;
        this->CMetricModel::createNewModels(n, m);
    }

    void test(core_t::TTime time) {
        m_ResourceMonitor.clearExtraMemory();
        this->createUpdateNewModels(time, m_ResourceMonitor);
    }

    std::size_t getNewPeople() const { return m_NewPeople; }

    std::size_t getNewAttributes() const { return m_NewAttributes; }

private:
    CResourceMonitor& m_ResourceMonitor;
    std::size_t m_NewPeople;
    std::size_t m_NewAttributes;
};

void addArrival(core_t::TTime time,
                const std::string& p,
                CDataGatherer& gatherer,
                CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&p);
    CEventData result;
    result.time(time);
    gatherer.addArrival(fields, result, resourceMonitor);
}

void addPersonData(std::size_t start,
                   std::size_t end,
                   core_t::TTime time,
                   CDataGatherer& gatherer,
                   CResourceMonitor& resourceMonitor) {
    for (std::size_t i = start; i < end; i++) {
        std::ostringstream ssA;
        ssA << "person" << i;
        addArrival(time, ssA.str(), gatherer, resourceMonitor);
    }
}

const std::string VALUE("23");

void addMetricArrival(core_t::TTime time,
                      const std::string& p,
                      CDataGatherer& gatherer,
                      CResourceMonitor& resourceMonitor) {
    CDataGatherer::TStrCPtrVec fields;
    fields.push_back(&p);
    fields.push_back(&VALUE);
    CEventData result;
    result.time(time);
    gatherer.addArrival(fields, result, resourceMonitor);
}

void addPersonMetricData(std::size_t start,
                         std::size_t end,
                         core_t::TTime time,
                         CDataGatherer& gatherer,
                         CResourceMonitor& resourceMonitor) {
    for (std::size_t i = start; i < end; i++) {
        std::ostringstream ssA;
        ssA << "person" << i;
        addMetricArrival(time, ssA.str(), gatherer, resourceMonitor);
    }
}

using TAddPersonDataFunc =
    std::function<void(std::size_t, std::size_t, core_t::TTime, CDataGatherer&, CResourceMonitor&)>;
using TMockModelInterfacePtr = std::shared_ptr<CMockModelInterface>;
using TModelFactoryPtr = std::shared_ptr<CModelFactory>;

TAddPersonDataFunc createModel(model_t::EModelType modelType,
                               const std::string& emptyString,
                               core_t::TTime firstTime,
                               core_t::TTime bucketLength,
                               CResourceMonitor& resourceMonitor,
                               TMockModelInterfacePtr& model,
                               CModelFactory::TDataGathererPtr& gatherer,
                               TModelFactoryPtr& factory) {
    switch (modelType) {
    case model_t::E_EventRateOnline: {
        // Test CEventRateModel::createUpdateNewModels()

        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);

        factory = std::make_shared<CEventRateModelFactory>(params, interimBucketCorrector);
        factory->identifier(1);
        factory->fieldNames(emptyString, emptyString, "pers", emptyString, TStrVec());
        CModelFactory::TFeatureVec features;
        features.push_back(model_t::E_IndividualCountByBucketAndPerson);
        factory->features(features);

        gatherer.reset(factory->makeDataGatherer(firstTime));

        const maths::CMultinomialConjugate conjugate;
        std::shared_ptr<::CMockEventRateModel> model_ = std::make_shared<::CMockEventRateModel>(
            factory->modelParams(), gatherer,
            factory->defaultFeatureModels(features, bucketLength, 0.4, true), conjugate,
            CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec(),
            resourceMonitor);

        BOOST_CHECK_EQUAL(model_t::E_EventRateOnline, model_->category());
        BOOST_TEST(model_->isPopulation() == false);

        model = model_;

        return addPersonData;
    }

    case model_t::E_MetricOnline: {
        SModelParams params(bucketLength);
        params.s_DecayRate = 0.001;
        auto interimBucketCorrector = std::make_shared<CInterimBucketCorrector>(bucketLength);
        factory = std::make_shared<CMetricModelFactory>(params, interimBucketCorrector);
        factory->identifier(1);
        factory->fieldNames(emptyString, emptyString, "peep", "val", TStrVec());
        factory->useNull(true);
        CModelFactory::TFeatureVec features;
        features.push_back(model_t::E_IndividualMeanByPerson);
        features.push_back(model_t::E_IndividualMinByPerson);
        features.push_back(model_t::E_IndividualMaxByPerson);
        factory->features(features);

        gatherer.reset(factory->makeDataGatherer(firstTime));

        std::shared_ptr<::CMockMetricModel> model_ = std::make_shared<::CMockMetricModel>(
            factory->modelParams(), gatherer,
            factory->defaultFeatureModels(features, bucketLength, 0.4, true),
            CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec(),
            resourceMonitor);

        BOOST_CHECK_EQUAL(model_t::E_MetricOnline, model_->category());
        BOOST_TEST(model_->isPopulation() == false);

        model = model_;
        return addPersonMetricData;
    }

    case model_t::E_Counting:
        return nullptr;
    }

    return nullptr;
}

struct SLargeAllocationTestParams {
    bool m_PersistInForeground;
    std::size_t m_MemoryLimit;
    std::size_t m_NumPeopleToBreachLimit;
    std::size_t m_NewPeopleLowerBound;
    std::size_t m_NewPeopleUpperBound;
    model_t::EModelType m_ModelType;
};

void doTestLargeAllocations(SLargeAllocationTestParams& param) {

    const std::string EMPTY_STRING("");
    const core_t::TTime FIRST_TIME(358556400);
    const core_t::TTime BUCKET_LENGTH(3600);

    CResourceMonitor resourceMonitor(param.m_PersistInForeground, 1.0);
    resourceMonitor.memoryLimit(std::size_t(param.m_MemoryLimit));

    TMockModelInterfacePtr model;
    CModelFactory::TDataGathererPtr gatherer;
    TModelFactoryPtr factory;
    TAddPersonDataFunc personAdder =
        createModel(param.m_ModelType, EMPTY_STRING, FIRST_TIME, BUCKET_LENGTH,
                    resourceMonitor, model, gatherer, factory);

    core_t::TTime time = FIRST_TIME;

    BOOST_TEST(resourceMonitor.areAllocationsAllowed());

    // Add some people & attributes to the gatherer
    // Run a sample
    // Check that the models can create the right number of people/attributes
    personAdder(0, 400, time, *gatherer, resourceMonitor);

    BOOST_CHECK_EQUAL(std::size_t(400), gatherer->numberActivePeople());

    LOG_DEBUG(<< "Testing for 1st time");
    model->test(time);
    BOOST_CHECK_EQUAL(std::size_t(400), gatherer->numberActivePeople());
    BOOST_CHECK_EQUAL(std::size_t(400), model->getNewPeople());
    BOOST_CHECK_EQUAL(std::size_t(0), model->getNewAttributes());
    time += BUCKET_LENGTH;

    personAdder(400, 1000, time, *gatherer, resourceMonitor);
    model->test(time);

    // This should add enough people to go over the memory limit
    personAdder(1000, param.m_NumPeopleToBreachLimit, time, *gatherer, resourceMonitor);

    LOG_DEBUG(<< "Testing for 2nd time");
    model->test(time);
    LOG_DEBUG(<< "# new people = " << model->getNewPeople());
    BOOST_TEST(model->getNewPeople() > param.m_NewPeopleLowerBound &&
                   model->getNewPeople() < param.m_NewPeopleUpperBound);
    BOOST_CHECK_EQUAL(std::size_t(0), model->getNewAttributes());
    BOOST_CHECK_EQUAL(model->getNewPeople(), gatherer->numberActivePeople());

    // Adding a small number of new people should be fine though,
    // as they're allowed in
    time += BUCKET_LENGTH;
    std::size_t oldNumberPeople{model->getNewPeople()};
    personAdder(param.m_NumPeopleToBreachLimit, param.m_NumPeopleToBreachLimit + 10,
                time, *gatherer, resourceMonitor);

    LOG_DEBUG(<< "Testing for 3rd time");
    model->test(time);
    BOOST_CHECK_EQUAL(oldNumberPeople + 10, model->getNewPeople());
    BOOST_CHECK_EQUAL(std::size_t(0), model->getNewAttributes());
    BOOST_CHECK_EQUAL(model->getNewPeople(), gatherer->numberActivePeople());
}
}

BOOST_AUTO_TEST_CASE(testLargeAllocations) {

    SLargeAllocationTestParams params[] = {
        {false, 70, 3000, 2700, 2900, model_t::E_EventRateOnline},
        {true, 70, 5000, 4500, 4700, model_t::E_EventRateOnline},
        {false, 100, 4000, 3500, 3700, model_t::E_MetricOnline},
        {true, 100, 7000, 6100, 6300, model_t::E_MetricOnline}};

    for (auto& param : params) {
        doTestLargeAllocations(param);
    }
}

void CResourceLimitTest::importCsvDataWithLimiter(core_t::TTime firstTime,
                                                  core_t::TTime bucketLength,
                                                  CResultWriter& outputResults,
                                                  const std::string& fileName,
                                                  CAnomalyDetector& detector,
                                                  std::size_t limitCutoff,
                                                  CResourceMonitor& resourceMonitor) {

    using TifstreamPtr = std::shared_ptr<std::ifstream>;
    TifstreamPtr ifs(new std::ifstream(fileName.c_str()));
    BOOST_TEST(ifs->is_open());

    core::CRegex regex;
    BOOST_TEST(regex.init(","));

    std::string line;
    // read the header
    BOOST_TEST(std::getline(*ifs, line));

    core_t::TTime lastBucketTime = firstTime;

    std::size_t i = 0;
    while (std::getline(*ifs, line)) {
        if (i == limitCutoff) {
            LOG_INFO(<< "Setting Limit cuttoff now");
            resourceMonitor.m_ByteLimitHigh = 0;
            resourceMonitor.m_ByteLimitLow = 0;
        }

        LOG_TRACE(<< "Got string: " << line);
        core::CRegex::TStrVec tokens;
        regex.split(line, tokens);

        core_t::TTime time;
        BOOST_TEST(core::CStringUtils::stringToType(tokens[0], time));

        for (/**/; lastBucketTime + bucketLength <= time; lastBucketTime += bucketLength) {
            outputResults(detector, lastBucketTime, lastBucketTime + bucketLength);
        }

        CAnomalyDetector::TStrCPtrVec fieldValues;
        for (std::size_t t = tokens.size() - 1; t > 0; t--) {
            fieldValues.push_back(&tokens[t]);
        }

        detector.addRecord(time, fieldValues);
        ++i;
    }

    outputResults(detector, lastBucketTime, lastBucketTime + bucketLength);

    ifs.reset();
}

BOOST_AUTO_TEST_SUITE_END()
