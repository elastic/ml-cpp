/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CBase64Filter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>

#include <maths/common/COrderings.h>
#include <maths/common/COrderingsSimultaneousSort.h>
#include <maths/common/CStatisticalTests.h>
#include <maths/common/CTools.h>
#include <maths/common/ProbabilityAggregators.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CAnomalyDetector.h>
#include <model/CAnomalyDetectorModel.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CCountingModel.h>
#include <model/CDataGatherer.h>
#include <model/CEventData.h>
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsAggregator.h>
#include <model/CHierarchicalResultsNormalizer.h>
#include <model/CHierarchicalResultsProbabilityFinalizer.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CLimits.h>
#include <model/CResourceMonitor.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "ModelTestHelpers.h"

#include <boost/algorithm/cxx11/is_sorted.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/unit_test.hpp>

#include <map>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <utility>

BOOST_AUTO_TEST_SUITE(CHierarchicalResultsTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TAttributeProbabilityVec = model::CHierarchicalResults::TAttributeProbabilityVec;
using TOptionalStr = std::optional<std::string>;
using TOptionalStrOptionalStrPr = model::CHierarchicalResults::TOptionalStrOptionalStrPr;

const std::string EMPTY_STRING;
const TOptionalStr EMPTY_OPTIONAL_STR;

//! \brief Checks that we visit the nodes in decreasing depth order.
class CBreadthFirstCheck final : public model::CHierarchicalResultsVisitor {
public:
    using TNodeCPtrSet = std::set<const TNode*>;
    using TNodeCPtrSetVec = std::vector<TNodeCPtrSet>;

public:
    CBreadthFirstCheck() = default;

    void visit(const model::CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) override {
        LOG_DEBUG(<< "Visiting " << node.print());

        if (node.s_Children.empty()) {
            // Leaf
            m_Layers[0].insert(&node);
            return;
        }

        // Check whether the children are on the layer below
        // otherwise start a new layer.

        std::size_t layer = m_Layer + 1;
        for (const auto* i : node.s_Children) {
            if (!m_Layers[m_Layer].contains(i)) {
                layer = m_Layer + 2;
                break;
            }
        }
        LOG_DEBUG(<< "layer = " << layer);

        m_Layer = layer - 1;
        if (layer > m_Layers.size() - 1) {
            m_Layers.resize(layer + 1);
        }
        m_Layers[layer].insert(&node);
    }

    void check(std::size_t expectedLayers) const {
        // Check we have the expected number of layers and that
        // all nodes are in a lower layer than their parents.

        LOG_DEBUG(<< "# layers = " << m_Layers.size());
        BOOST_REQUIRE_EQUAL(expectedLayers, m_Layers.size());

        for (std::size_t i = 0; i < m_Layers.size(); ++i) {
            LOG_DEBUG(<< "Checking layer " << m_Layers[i]);
            for (auto itr = m_Layers[i].begin(); itr != m_Layers[i].end(); ++itr) {
                if ((*itr)->s_Parent != nullptr) {
                    std::size_t p = this->layer((*itr)->s_Parent);
                    LOG_DEBUG(<< "layer = " << i << ", parent layer = " << p);
                    BOOST_TEST_REQUIRE(p > i);
                }
            }
        }
    }

private:
    //! Get a node's layer.
    std::size_t layer(const TNode* node) const {
        for (std::size_t i = 0; i < m_Layers.size(); ++i) {
            if (m_Layers[i].contains(node)) {
                return i;
            }
        }

        LOG_ERROR(<< "Couldn't find node " << node->print());
        BOOST_TEST_REQUIRE(false);

        return 0;
    }

private:
    std::size_t m_Layer = 0;
    TNodeCPtrSetVec m_Layers{1, TNodeCPtrSet()};
};

//! \brief Checks that we visit all a nodes children immediately
//! before visiting it.
class CDepthFirstCheck : public model::CHierarchicalResultsVisitor {
public:
    using TNodeCPtrVec = std::vector<const TNode*>;

public:
    void visit(const model::CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) override {
        LOG_DEBUG(<< "Visiting " << node.print());
        for (std::size_t i = node.s_Children.size(); i > 0; --i) {
            BOOST_TEST_REQUIRE(!m_Children.empty());
            BOOST_REQUIRE_EQUAL(m_Children.back(), node.s_Children[i - 1]);
            m_Children.pop_back();
        }
        m_Children.push_back(&node);
    }

private:
    TNodeCPtrVec m_Children;
};

//! \brief A pretty print of the hierarchical results.
class CPrinter : public model::CHierarchicalResultsVisitor {
public:
    CPrinter() : m_ShouldPrintWrittenNodesOnly(false) {}

    explicit CPrinter(bool shouldOnlyPrintWrittenNodes)
        : m_ShouldPrintWrittenNodesOnly(shouldOnlyPrintWrittenNodes) {}

    void visit(const model::CHierarchicalResults& results, const TNode& node, bool pivot) override {
        if (m_ShouldPrintWrittenNodesOnly == false ||
            shouldWriteResult(m_Limits, results, node, pivot)) {
            m_Result = std::string(2 * depth(&node), ' ') + node.print() +
                       (pivot ? " pivot" : "") + (m_Result.empty() ? "" : "\n") + m_Result;
        }
    }

    const std::string& result() const { return m_Result; }

private:
    static std::size_t depth(const TNode* node) {
        std::size_t result = 0;
        for (/**/; node->s_Parent != nullptr; node = node->s_Parent) {
            ++result;
        }
        return result;
    }

private:
    bool m_ShouldPrintWrittenNodesOnly;
    std::string m_Result;
    model::CLimits m_Limits;
};

//! \brief Gets the various types of nodes.
class CNodeExtractor : public model::CHierarchicalResultsVisitor {
public:
    using TNodeCPtrVec = std::vector<const TNode*>;

public:
    void visit(const model::CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) override {
        if (isPartitioned(node)) {
            m_PartitionedNodes.push_back(&node);
        }
        if (isPartition(node)) {
            m_PartitionNodes.push_back(&node);
        }
        if (isPerson(node)) {
            m_PersonNodes.push_back(&node);
        }
        if (isLeaf(node)) {
            m_LeafNodes.push_back(&node);
        }
    }

    const TNodeCPtrVec& partitionedNodes() const { return m_PartitionedNodes; }
    const TNodeCPtrVec& partitionNodes() const { return m_PartitionNodes; }
    const TNodeCPtrVec& personNodes() const { return m_PersonNodes; }
    const TNodeCPtrVec& leafNodes() const { return m_LeafNodes; }

private:
    TNodeCPtrVec m_PartitionedNodes;
    TNodeCPtrVec m_PartitionNodes;
    TNodeCPtrVec m_PersonNodes;
    TNodeCPtrVec m_LeafNodes;
};

//! \brief Checks our anomaly scores are correct post scoring.
class CCheckScores : public model::CHierarchicalResultsVisitor {
public:
    void visit(const model::CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) override {
        LOG_DEBUG(<< node.s_Spec.print() << " score = " << node.s_RawAnomalyScore << ", expected score = "
                  << maths::common::CTools::anomalyScore(node.probability()));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(maths::common::CTools::anomalyScore(node.probability()),
                                     node.s_RawAnomalyScore, 1e-10);
    }
};

//! \brief Checks that if we write a result for a node, we also write one
//! for its parent (if there is one) and one for at least one child (if
//! there are any children).
class CWriteConsistencyChecker final : public model::CHierarchicalResultsVisitor {
public:
    explicit CWriteConsistencyChecker(const model::CLimits& limits)
        : m_Limits(limits) {}

    void visit(const model::CHierarchicalResults& results, const TNode& node, bool pivot) override {
        if (!CHierarchicalResultsTest::CWriteConsistencyChecker::shouldWriteResult(
                m_Limits, results, node, pivot)) {
            return;
        }
        if (!CHierarchicalResultsTest::CWriteConsistencyChecker::isLeaf(node)) {
            bool willWriteAChild(false);
            for (const auto* i : node.s_Children) {
                BOOST_TEST_REQUIRE(i != nullptr);
                willWriteAChild = CHierarchicalResultsTest::CWriteConsistencyChecker::shouldWriteResult(
                    m_Limits, results, *i, pivot);
                if (willWriteAChild) {
                    break;
                }
            }
            BOOST_TEST_REQUIRE(willWriteAChild);
        }

        if (!CHierarchicalResultsTest::CWriteConsistencyChecker::isRoot(node)) {
            BOOST_TEST_REQUIRE(node.s_Parent != nullptr);
            if (isTypeForWhichWeWriteResults(*node.s_Parent, pivot)) {
                BOOST_TEST_REQUIRE(this->shouldWriteResult(m_Limits, results,
                                                           *node.s_Parent, pivot));
            }
        }
    }

private:
    const model::CLimits& m_Limits;
};

using TIntDoubleVecMap = std::map<int, TDoubleVec>;

//! \brief Node probability container.
struct SNodeProbabilities {
    explicit SNodeProbabilities(std::string name) : s_Name(std::move(name)) {}

    std::string s_Name;
    TIntDoubleVecMap s_Probabilities;
};

//! \brief Gathers detector probabilities by level.
class CProbabilityGatherer final
    : public model::CHierarchicalResultsLevelSet<SNodeProbabilities> {
public:
    using TBase = CHierarchicalResultsLevelSet<SNodeProbabilities>;
    using TNodeProbabilitiesPtrVec = TTypePtrVec;

    class CFactory {
    public:
        static SNodeProbabilities
        make(const model::CHierarchicalResults::TNode& node, bool /*unused*/) {
            return SNodeProbabilities(node.s_Spec.s_PartitionFieldName.value_or("") + ' ' +
                                      node.s_Spec.s_PersonFieldName.value_or("") + ' ' +
                                      node.s_Spec.s_FunctionName.value_or("") + ' ' +
                                      node.s_Spec.s_ValueFieldName.value_or(""));
        }
    };

public:
    CProbabilityGatherer() : TBase(SNodeProbabilities("bucket")) {}

    void visit(const model::CHierarchicalResults& /*results*/, const TNode& node, bool pivot) override {
        if (isLeaf(node)) {
            constexpr CFactory factory;
            TNodeProbabilitiesPtrVec probabilities;
            this->elements(node, pivot, factory, probabilities);
            for (const auto& probability : probabilities) {
                if (node.probability() <
                    model::CDetectorEqualizer::largestProbabilityToCorrect()) {
                    probability->s_Probabilities[node.s_Detector].push_back(node.probability());
                }
            }
        }
    }

    double test(double minimumSignificance) const {
        maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator meanSignificance;

        for (const auto & [ _, probabilities ] : this->leafSet()) {
            LOG_DEBUG(<< "leaf = " << probabilities.s_Name);

            std::vector<int> detectors;
            for (const auto & [ detector, p ] : probabilities.s_Probabilities) {
                detectors.push_back(detector);
            }

            for (std::size_t j = 1; j < detectors.size(); ++j) {
                for (std::size_t k = 0; k < j; ++k) {
                    double significance = maths::common::CStatisticalTests::twoSampleKS(
                        probabilities.s_Probabilities.find(detectors[j])->second,
                        probabilities.s_Probabilities.find(detectors[k])->second);
                    LOG_DEBUG(<< detectors[j] << " vs " << detectors[k]
                              << ": significance = " << significance);
                    BOOST_TEST_REQUIRE(significance > minimumSignificance);
                    meanSignificance.add(std::log(significance));
                }
            }
        }

        return std::exp(maths::common::CBasicStatistics::mean(meanSignificance));
    }
};

//! Compute the probability of the samples [\p begin, \p end).
template<typename ITR>
void addAggregateValues(double w1,
                        double w2,
                        std::size_t n,
                        ITR begin,
                        ITR end,
                        TDoubleVec& scores,
                        TDoubleVec& probabilities) {
    double score;
    double probability;
    TDoubleVec const probs(begin, end);
    model::CAnomalyScore::compute(w1, w2, 1, n, 0.05, probs, score, probability);
    scores.push_back(score);
    probabilities.push_back(probability);
}

void addResult(int detector,
               bool isPopulation,
               const std::string& functionName,
               ml::model::function_t::EFunction function,
               const std::string& partitionFieldName,
               const std::string& partitionFieldValue,
               const std::string& personFieldName,
               const std::string& personFieldValue,
               const std::string& valueFieldName,
               double p,
               ml::model::CHierarchicalResults& results) {
    ml::model::SAnnotatedProbability annotatedProbability(p);
    results.addModelResult(detector, isPopulation, functionName, function,
                           partitionFieldName, partitionFieldValue, personFieldName,
                           personFieldValue, valueFieldName, annotatedProbability);
}

void addResult(int detector,
               bool isPopulation,
               const std::string& functionName,
               ml::model::function_t::EFunction function,
               const std::string& partitionFieldName,
               const std::string& partitionFieldValue,
               const std::string& personFieldName,
               const std::string& personFieldValue,
               const std::string& valueFieldName,
               double p,
               const ml::model::CAnomalyDetectorModel* model,
               ml::model::CHierarchicalResults& results) {
    ml::model::SAnnotatedProbability annotatedProbability(p);
    results.addModelResult(detector, isPopulation, functionName, function, partitionFieldName,
                           partitionFieldValue, personFieldName, personFieldValue,
                           valueFieldName, annotatedProbability, model);
}

} // unnamed::

BOOST_AUTO_TEST_CASE(testBreadthFirstVisit) {
    model::CHierarchicalResults results;

    // Three partitioning fields PART1, PART2, PART3:
    //   - Two partitions for PART1 part1 and part2
    //   - Two partitions for PART2 part1 and part2
    //   - One partitions for PART3 part1
    //
    // One person field PERS with three values pers1, pers2, pers3 and pers4
    //
    // Two value fields VAL1 and VAL2

    static const std::string PART1("PART1");
    static const std::string PART2("PART2");
    static const std::string PART3("PART3");
    static const std::string part1("PART1");
    static const std::string part2("PART1");

    static const std::string PERS("PERS");
    std::string const pers1("pers1");
    std::string const pers2("pers2");
    std::string const pers3("pers3");
    std::string const pers4("pers4");

    static const std::string VAL1("VAL1");
    static const std::string VAL2("VAL1");

    static const std::string FUNC("min");

    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMin);

    addResult(1, false, FUNC, function, PART1, part1, PERS, pers1, VAL1, 0.1, results);
    addResult(1, false, FUNC, function, PART1, part1, PERS, pers2, VAL1, 0.1, results);
    addResult(1, false, FUNC, function, PART1, part1, PERS, pers3, VAL1, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part1, PERS, pers1, EMPTY_STRING, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part1, PERS, pers2, EMPTY_STRING, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part1, PERS, pers3, EMPTY_STRING, 0.1, results);
    addResult(3, false, FUNC, function, PART1, part1, PERS, pers1, VAL2, 0.1, results);
    addResult(3, false, FUNC, function, PART1, part1, PERS, pers2, VAL2, 0.1, results);
    addResult(3, false, FUNC, function, PART1, part1, PERS, pers4, VAL2, 0.1, results);
    addResult(1, false, FUNC, function, PART1, part2, PERS, pers1, VAL1, 0.1, results);
    addResult(1, false, FUNC, function, PART1, part2, PERS, pers2, VAL1, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part2, PERS, pers1, EMPTY_STRING, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part2, PERS, pers3, EMPTY_STRING, 0.1, results);
    addResult(3, false, FUNC, function, PART1, part2, PERS, pers4, VAL2, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part1, PERS, pers1, VAL1, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part1, PERS, pers2, VAL1, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part2, PERS, pers1, VAL1, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part2, PERS, pers2, VAL1, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part2, PERS, pers3, VAL1, 0.1, results);
    addResult(5, false, FUNC, function, PART2, part2, PERS, pers1, VAL2, 0.1, results);
    addResult(5, false, FUNC, function, PART2, part2, PERS, pers2, VAL2, 0.1, results);
    addResult(5, false, FUNC, function, PART2, part2, PERS, pers3, VAL2, 0.1, results);
    addResult(6, true, FUNC, function, PART3, part1, PERS, pers1, VAL1, 0.1, results);
    addResult(6, true, FUNC, function, PART3, part1, PERS, pers2, VAL1, 0.1, results);

    results.buildHierarchy();

    CBreadthFirstCheck bfc;
    results.bottomUpBreadthFirst(bfc);
    bfc.check(5 /*expected layers*/);
}

BOOST_AUTO_TEST_CASE(testDepthFirstVisit) {
    model::CHierarchicalResults results;

    // Three partitioning fields PART1, PART2, PART3:
    //   - Two partitions for PART1 part1 and part2
    //   - Two partitions for PART2 part1 and part2
    //   - One partitions for PART2 part1
    //
    // One person field PERS with three values pers1, pers2, pers3 and pers4
    //
    // Two value fields VAL1 and VAL2

    static const std::string PART1("PART1");
    static const std::string PART2("PART2");
    static const std::string PART3("PART3");
    std::string const part1("PART1");
    std::string const part2("PART1");

    static const std::string PERS("PERS");
    std::string const pers1("pers1");
    std::string const pers2("pers2");
    std::string const pers3("pers3");
    std::string const pers4("pers4");

    static const std::string VAL1("VAL1");
    static const std::string VAL2("VAL1");

    static const std::string FUNC("max");

    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMax);

    addResult(1, false, FUNC, function, PART1, part1, PERS, pers1, VAL1, 0.1, results);
    addResult(1, false, FUNC, function, PART1, part1, PERS, pers2, VAL1, 0.1, results);
    addResult(1, false, FUNC, function, PART1, part1, PERS, pers3, VAL1, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part1, PERS, pers1, EMPTY_STRING, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part1, PERS, pers2, EMPTY_STRING, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part1, PERS, pers3, EMPTY_STRING, 0.1, results);
    addResult(3, false, FUNC, function, PART1, part1, PERS, pers1, VAL2, 0.1, results);
    addResult(3, false, FUNC, function, PART1, part1, PERS, pers2, VAL2, 0.1, results);
    addResult(3, false, FUNC, function, PART1, part1, PERS, pers4, VAL2, 0.1, results);
    addResult(1, false, FUNC, function, PART1, part2, PERS, pers1, VAL1, 0.1, results);
    addResult(1, false, FUNC, function, PART1, part2, PERS, pers2, VAL1, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part2, PERS, pers1, EMPTY_STRING, 0.1, results);
    addResult(2, true, FUNC, function, PART1, part2, PERS, pers3, EMPTY_STRING, 0.1, results);
    addResult(3, false, FUNC, function, PART1, part2, PERS, pers4, VAL2, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part1, PERS, pers1, VAL1, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part1, PERS, pers2, VAL1, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part2, PERS, pers1, VAL1, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part2, PERS, pers2, VAL1, 0.1, results);
    addResult(4, false, FUNC, function, PART2, part2, PERS, pers3, VAL1, 0.1, results);
    addResult(5, false, FUNC, function, PART2, part2, PERS, pers1, VAL2, 0.1, results);
    addResult(5, false, FUNC, function, PART2, part2, PERS, pers2, VAL2, 0.1, results);
    addResult(5, false, FUNC, function, PART2, part2, PERS, pers3, VAL2, 0.1, results);
    addResult(6, true, FUNC, function, PART3, part1, PERS, pers1, VAL1, 0.1, results);
    addResult(6, true, FUNC, function, PART3, part1, PERS, pers2, VAL1, 0.1, results);

    results.buildHierarchy();

    CDepthFirstCheck dfc;
    results.postorderDepthFirst(dfc);
}

namespace {

const std::string FALSE_STR("false");
const std::string TRUE_STR("true");

const std::string PNF1("PNF1");
const std::string pn11("pn11");
const std::string pn12("pn12");
const std::string pn13("pn13");

const std::string PNF2("PNF2");
const std::string pn21("pn21");
const std::string pn22("pn22");
const std::string pn23("pn23");

const std::string PF1("PF1");
const std::string p11("p11");
const std::string p12("p12");
const std::string p13("p13");
const std::string p14("p14");
const std::string p15("p15");
const std::string p16("p16");

const std::string PF2("PF2");
const std::string p21("p21");
const std::string p22("p22");
const std::string p23("p23");

} // unnamed::

BOOST_AUTO_TEST_CASE(testBuildHierarchy) {
    static const std::string FUNC("mean");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMean);

    // Test vanilla by / over.
    {
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 1.0, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        BOOST_REQUIRE_EQUAL(std::string("'false/false/mean/////': 1, 0"),
                            printer.result());
    }
    {
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  p11, EMPTY_STRING, 0.01, results);
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  p12, EMPTY_STRING, 0.03, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        BOOST_REQUIRE_EQUAL(std::string("'false/false////PF1//': 1, 0\n"
                                        "  'false/false/mean///PF1/p12/': 0.03, 0\n"
                                        "  'false/false/mean///PF1/p11/': 0.01, 0"),
                            printer.result());
    }
    {
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 0.3, results);
        addResult(2, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.01, results);
        addResult(2, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1, p12,
                  EMPTY_STRING, 0.03, results);
        addResult(3, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF2,
                  p22, EMPTY_STRING, 0.03, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nover:\n" << printer.result());
        BOOST_REQUIRE_EQUAL(std::string("'false/true//////': 1, 0\n"
                                        "  'false/false/mean///PF2/p22/': 0.03, 0\n"
                                        "  'false/true////PF1//': 1, 0\n"
                                        "    'false/true/mean///PF1/p12/': 0.03, 0\n"
                                        "    'false/true/mean///PF1/p11/': 0.01, 0\n"
                                        "  'false/false/mean/////': 0.3, 0"),
                            printer.result());
    }

    // Test vanilla partition
    {
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, PNF1, pn11, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(1, false, FUNC, function, PNF1, pn12, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(1, false, FUNC, function, PNF1, pn13, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.05, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\npartition:\n" << printer.result());
        BOOST_REQUIRE_EQUAL(std::string("'false/false//PNF1////': 1, 0\n"
                                        "  'false/false/mean/PNF1/pn13///': 0.05, 0\n"
                                        "  'false/false/mean/PNF1/pn12///': 0.01, 0\n"
                                        "  'false/false/mean/PNF1/pn11///': 0.01, 0"),
                            printer.result());
    }

    // Test complex.
    {
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(2, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  p11, EMPTY_STRING, 0.01, results);
        addResult(2, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  p14, EMPTY_STRING, 0.01, results);
        addResult(3, false, FUNC, function, PNF1, pn11, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(3, false, FUNC, function, PNF1, pn12, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(3, false, FUNC, function, PNF1, pn13, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.05, results);
        addResult(4, true, FUNC, function, PNF2, pn22, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(4, true, FUNC, function, PNF2, pn23, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.05, results);
        addResult(5, true, FUNC, function, PNF2, pn21, PF1, p11, EMPTY_STRING, 0.2, results);
        addResult(5, true, FUNC, function, PNF2, pn22, PF1, p11, EMPTY_STRING, 0.2, results);
        addResult(5, true, FUNC, function, PNF2, pn22, PF1, p12, EMPTY_STRING, 0.1, results);
        addResult(6, true, FUNC, function, PNF2, pn22, PF2, p21, EMPTY_STRING, 0.15, results);
        addResult(7, false, FUNC, function, PNF2, pn22, PF2, p21, EMPTY_STRING, 0.12, results);
        addResult(6, true, FUNC, function, PNF2, pn22, PF2, p23, EMPTY_STRING, 0.12, results);
        addResult(7, false, FUNC, function, PNF2, pn22, PF2, p23, EMPTY_STRING, 0.82, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\ncomplex:\n" << printer.result());
        BOOST_REQUIRE_EQUAL(
            std::string("'false/true//////': 1, 0\n"
                        "  'false/true//PNF2////': 1, 0\n"
                        "    'false/true/mean/PNF2/pn23///': 0.05, 0\n"
                        "    'false/true//PNF2/pn22///': 1, 0\n"
                        "      'false/true//PNF2/pn22/PF2//': 1, 0\n"
                        "        'false/true/mean/PNF2/pn22/PF2/p23/': 0.12, 0\n"
                        "        'false/false/mean/PNF2/pn22/PF2/p23/': 0.82, 0\n"
                        "        'false/true/mean/PNF2/pn22/PF2/p21/': 0.15, 0\n"
                        "        'false/false/mean/PNF2/pn22/PF2/p21/': 0.12, 0\n"
                        "      'false/true//PNF2/pn22/PF1//': 1, 0\n"
                        "        'false/true/mean/PNF2/pn22/PF1/p12/': 0.1, 0\n"
                        "        'false/true/mean/PNF2/pn22/PF1/p11/': 0.2, 0\n"
                        "      'false/true/mean/PNF2/pn22///': 0.01, 0\n"
                        "    'false/true/mean/PNF2/pn21/PF1/p11/': 0.2, 0\n"
                        "  'false/false//PNF1////': 1, 0\n"
                        "    'false/false/mean/PNF1/pn13///': 0.05, 0\n"
                        "    'false/false/mean/PNF1/pn12///': 0.01, 0\n"
                        "    'false/false/mean/PNF1/pn11///': 0.01, 0\n"
                        "  'false/false//////': 1, 0\n"
                        "    'false/false////PF1//': 1, 0\n"
                        "      'false/false/mean///PF1/p14/': 0.01, 0\n"
                        "      'false/false/mean///PF1/p11/': 0.01, 0\n"
                        "    'false/false/mean/////': 0.01, 0"),
            printer.result());
    }
}

BOOST_AUTO_TEST_CASE(testBuildHierarchyGivenPartitionsWithSinglePersonFieldValue) {
    static const std::string FUNC("mean");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMean);

    std::string const partition("par");
    std::string const partition1("par_1");
    std::string const partition2("par_2");
    std::string const person("p");
    std::string const person1("p_1");
    std::string const valueField("value");

    model::CHierarchicalResults results;
    addResult(1, false, FUNC, function, partition, partition1, person, person1,
              valueField, 0.01, results);
    addResult(1, false, FUNC, function, partition, partition2, person, person1,
              valueField, 0.01, results);
    results.buildHierarchy();

    CNodeExtractor extract;
    results.bottomUpBreadthFirst(extract);
    BOOST_REQUIRE_EQUAL(1, extract.partitionedNodes().size());
    BOOST_REQUIRE_EQUAL(2, extract.partitionNodes().size());
    BOOST_REQUIRE_EQUAL(2, extract.personNodes().size());

    // partitioned node
    BOOST_REQUIRE_EQUAL(partition, *extract.partitionedNodes()[0]->s_Spec.s_PartitionFieldName);
    BOOST_REQUIRE_EQUAL(
        EMPTY_STRING,
        extract.partitionedNodes()[0]->s_Spec.s_PartitionFieldValue.value_or(""));
    BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                        extract.partitionedNodes()[0]->s_Spec.s_PersonFieldName.value_or(""));
    BOOST_REQUIRE_EQUAL(
        EMPTY_STRING,
        extract.partitionedNodes()[0]->s_Spec.s_PersonFieldValue.value_or(""));

    // partition nodes
    BOOST_REQUIRE_EQUAL(partition, *extract.partitionNodes()[0]->s_Spec.s_PartitionFieldName);
    BOOST_REQUIRE_EQUAL(partition1, *extract.partitionNodes()[0]->s_Spec.s_PartitionFieldValue);
    BOOST_REQUIRE_EQUAL(person, *extract.partitionNodes()[0]->s_Spec.s_PersonFieldName);
    BOOST_REQUIRE_EQUAL(person1, *extract.partitionNodes()[0]->s_Spec.s_PersonFieldValue);

    BOOST_REQUIRE_EQUAL(partition, *extract.partitionNodes()[1]->s_Spec.s_PartitionFieldName);
    BOOST_REQUIRE_EQUAL(partition2, *extract.partitionNodes()[1]->s_Spec.s_PartitionFieldValue);
    BOOST_REQUIRE_EQUAL(person, *extract.partitionNodes()[1]->s_Spec.s_PersonFieldName);
    BOOST_REQUIRE_EQUAL(person1, *extract.partitionNodes()[1]->s_Spec.s_PersonFieldValue);

    // person nodes
    BOOST_REQUIRE_EQUAL(partition, *extract.personNodes()[0]->s_Spec.s_PartitionFieldName);
    BOOST_REQUIRE_EQUAL(partition1, *extract.personNodes()[0]->s_Spec.s_PartitionFieldValue);
    BOOST_REQUIRE_EQUAL(person, *extract.personNodes()[0]->s_Spec.s_PersonFieldName);
    BOOST_REQUIRE_EQUAL(person1, *extract.personNodes()[0]->s_Spec.s_PersonFieldValue);
    BOOST_REQUIRE_EQUAL(0, extract.personNodes()[0]->s_Children.size());

    BOOST_REQUIRE_EQUAL(partition, *extract.personNodes()[1]->s_Spec.s_PartitionFieldName);
    BOOST_REQUIRE_EQUAL(partition2, *extract.personNodes()[1]->s_Spec.s_PartitionFieldValue);
    BOOST_REQUIRE_EQUAL(person, *extract.personNodes()[1]->s_Spec.s_PersonFieldName);
    BOOST_REQUIRE_EQUAL(person1, *extract.personNodes()[1]->s_Spec.s_PersonFieldValue);
    BOOST_REQUIRE_EQUAL(0, extract.personNodes()[1]->s_Children.size());
}

BOOST_AUTO_TEST_CASE(testBasicVisitor) {
    static const std::string FUNC("max");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMax);

    // Test by and over
    {
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 1.0, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        CNodeExtractor extract;
        results.bottomUpBreadthFirst(extract);
        BOOST_REQUIRE_EQUAL(0, extract.partitionedNodes().size());
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes().size());
        BOOST_REQUIRE_EQUAL(0, extract.personNodes().size());
    }
    {
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  EMPTY_STRING, EMPTY_STRING, 1.0, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        CNodeExtractor extract;
        results.bottomUpBreadthFirst(extract);
        BOOST_REQUIRE_EQUAL(0, extract.partitionedNodes().size());
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes().size());
        BOOST_REQUIRE_EQUAL(1, extract.personNodes().size());
        BOOST_REQUIRE_EQUAL(PF1, *extract.personNodes()[0]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.personNodes()[0]->s_Spec.s_PersonFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[0]->s_Children.size());
    }
    {
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  EMPTY_STRING, EMPTY_STRING, 1.0, results);
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  p11, EMPTY_STRING, 1.0, results);
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  p12, EMPTY_STRING, 1.0, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        CNodeExtractor extract;
        results.bottomUpBreadthFirst(extract);

        BOOST_REQUIRE_EQUAL(0, extract.partitionedNodes().size());
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes().size());
        BOOST_REQUIRE_EQUAL(1, extract.personNodes().size());
        BOOST_REQUIRE_EQUAL(3, extract.leafNodes().size());
        BOOST_REQUIRE_EQUAL(FUNC, *extract.leafNodes()[0]->s_Spec.s_FunctionName);
        BOOST_REQUIRE_EQUAL(FUNC, *extract.leafNodes()[1]->s_Spec.s_FunctionName);
        BOOST_REQUIRE_EQUAL(FUNC, *extract.leafNodes()[2]->s_Spec.s_FunctionName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.leafNodes()[0]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.leafNodes()[1]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.leafNodes()[2]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.leafNodes()[0]->s_Spec.s_PersonFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(p11, *extract.leafNodes()[1]->s_Spec.s_PersonFieldValue);
        BOOST_REQUIRE_EQUAL(p12, *extract.leafNodes()[2]->s_Spec.s_PersonFieldValue);
        BOOST_REQUIRE_EQUAL(0, extract.leafNodes()[0]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.leafNodes()[1]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.leafNodes()[2]->s_Children.size());
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.personNodes()[0]->s_Spec.s_FunctionName.value_or(""));
        BOOST_REQUIRE_EQUAL(PF1, *extract.personNodes()[0]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.personNodes()[0]->s_Spec.s_PersonFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(3, extract.personNodes()[0]->s_Children.size());
    }
    {

        model::CHierarchicalResults results;
        addResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  EMPTY_STRING, EMPTY_STRING, 1.0, results);
        addResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 1.0, results);
        addResult(2, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF2, p23,
                  EMPTY_STRING, 1.0, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nover:\n" << printer.result());
        CNodeExtractor extract;
        results.bottomUpBreadthFirst(extract);
        BOOST_REQUIRE_EQUAL(0, extract.partitionedNodes().size());
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes().size());
        BOOST_REQUIRE_EQUAL(3, extract.personNodes().size());
        BOOST_REQUIRE_EQUAL(FUNC, *extract.personNodes()[0]->s_Spec.s_FunctionName);
        BOOST_REQUIRE_EQUAL(FUNC, *extract.personNodes()[1]->s_Spec.s_FunctionName);
        BOOST_REQUIRE_EQUAL(FUNC, *extract.personNodes()[2]->s_Spec.s_FunctionName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.personNodes()[0]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.personNodes()[1]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF2, *extract.personNodes()[2]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.personNodes()[0]->s_Spec.s_PersonFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(p11, *extract.personNodes()[1]->s_Spec.s_PersonFieldValue);
        BOOST_REQUIRE_EQUAL(p23, *extract.personNodes()[2]->s_Spec.s_PersonFieldValue);
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[0]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[1]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[2]->s_Children.size());
    }
    {
        LOG_DEBUG(<< "Clear...");
        model::CHierarchicalResults results;
        addResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.2, results);
        addResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.3, results);
        addResult(2, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF2,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(3, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 1.0, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nover:\n" << printer.result());
        CNodeExtractor extract;
        results.bottomUpBreadthFirst(extract);
        BOOST_REQUIRE_EQUAL(0, extract.partitionedNodes().size());
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes().size());
        BOOST_REQUIRE_EQUAL(2, extract.personNodes().size());
        BOOST_REQUIRE_EQUAL(FUNC, *extract.personNodes()[0]->s_Spec.s_FunctionName);
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.personNodes()[1]->s_Spec.s_FunctionName.value_or(""));
        BOOST_REQUIRE_EQUAL(PF2, *extract.personNodes()[0]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.personNodes()[1]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.personNodes()[0]->s_Spec.s_PersonFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(p11, *extract.personNodes()[1]->s_Spec.s_PersonFieldValue);
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[0]->s_Children.size());
        BOOST_REQUIRE_EQUAL(2, extract.personNodes()[1]->s_Children.size());
    }

    // Test partition
    {
        model::CHierarchicalResults results;
        addResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.2, results);
        addResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.3, results);
        addResult(2, true, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF2,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(3, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 1.0, results);
        addResult(4, true, FUNC, function, PNF1, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.2, results);
        addResult(4, true, FUNC, function, PNF1, pn11, PF1, p11, EMPTY_STRING, 0.3, results);
        addResult(5, true, FUNC, function, PNF1, pn12, PF2, EMPTY_STRING,
                  EMPTY_STRING, 0.01, results);
        addResult(6, true, FUNC, function, PNF1, pn13, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 1.0, results);
        results.buildHierarchy();
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\npartition:\n" << printer.result());
        CNodeExtractor extract;
        results.bottomUpBreadthFirst(extract);

        BOOST_REQUIRE_EQUAL(1, extract.partitionedNodes().size());
        BOOST_REQUIRE_EQUAL(PNF1, *extract.partitionedNodes()[0]->s_Spec.s_PartitionFieldName);
        BOOST_REQUIRE_EQUAL(
            EMPTY_STRING,
            extract.partitionedNodes()[0]->s_Spec.s_PartitionFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(4, extract.partitionedNodes()[0]->s_Children.size());

        BOOST_REQUIRE_EQUAL(4, extract.partitionNodes().size());
        BOOST_REQUIRE_EQUAL(PNF1, *extract.partitionNodes()[0]->s_Spec.s_PartitionFieldName);
        BOOST_REQUIRE_EQUAL(PNF1, *extract.partitionNodes()[1]->s_Spec.s_PartitionFieldName);
        BOOST_REQUIRE_EQUAL(PNF1, *extract.partitionNodes()[2]->s_Spec.s_PartitionFieldName);
        BOOST_REQUIRE_EQUAL(PNF1, *extract.partitionNodes()[3]->s_Spec.s_PartitionFieldName);
        BOOST_REQUIRE_EQUAL(
            EMPTY_STRING,
            extract.partitionNodes()[0]->s_Spec.s_PartitionFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(pn11, *extract.partitionNodes()[1]->s_Spec.s_PartitionFieldValue);
        BOOST_REQUIRE_EQUAL(pn12, *extract.partitionNodes()[2]->s_Spec.s_PartitionFieldValue);
        BOOST_REQUIRE_EQUAL(pn13, *extract.partitionNodes()[3]->s_Spec.s_PartitionFieldValue);
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes()[0]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes()[1]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes()[2]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.partitionNodes()[3]->s_Children.size());

        BOOST_REQUIRE_EQUAL(5, extract.personNodes().size());
        BOOST_REQUIRE_EQUAL(PF2, *extract.personNodes()[0]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.personNodes()[1]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.personNodes()[2]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF2, *extract.personNodes()[3]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(PF1, *extract.personNodes()[4]->s_Spec.s_PersonFieldName);
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.personNodes()[0]->s_Spec.s_PersonFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(p11, *extract.personNodes()[1]->s_Spec.s_PersonFieldValue);
        BOOST_REQUIRE_EQUAL(p11, *extract.personNodes()[2]->s_Spec.s_PersonFieldValue);
        BOOST_REQUIRE_EQUAL(EMPTY_STRING,
                            extract.personNodes()[3]->s_Spec.s_PersonFieldValue.value_or(""));
        BOOST_REQUIRE_EQUAL(p11, *extract.personNodes()[4]->s_Spec.s_PersonFieldValue);
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[0]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[1]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[2]->s_Children.size());
        BOOST_REQUIRE_EQUAL(0, extract.personNodes()[3]->s_Children.size());
        BOOST_REQUIRE_EQUAL(2, extract.personNodes()[4]->s_Children.size());
    }
}

BOOST_AUTO_TEST_CASE(testAggregator) {
    using TAnnotatedProbabilityVec = std::vector<model::SAnnotatedProbability>;

    model::CAnomalyDetectorModelConfig const modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    model::CHierarchicalResultsAggregator aggregator(modelConfig);
    model::CAnomalyScore::CComputer const attributeComputer(
        0.5, 0.5, 1, 5, modelConfig.maximumAnomalousProbability());
    model::CAnomalyScore::CComputer const personComputer(
        0.0, 1.0, 1, 1, modelConfig.maximumAnomalousProbability());
    model::CAnomalyScore::CComputer const partitionComputer(
        0.0, 1.0, 1, 1, modelConfig.maximumAnomalousProbability());
    double score = 0.0;
    double probability = 1.0;
    static const std::string FUNC("max");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMax);

    // Test by.
    {
        TDoubleVec const probabilities{0.22, 0.03, 0.02};
        TAnnotatedProbabilityVec annotatedProbabilities;
        for (auto p : probabilities) {
            annotatedProbabilities.emplace_back(p);
        }

        model::CHierarchicalResults results;
        results.addModelResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p11, EMPTY_STRING, annotatedProbabilities[0]);
        results.addModelResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p12, EMPTY_STRING, annotatedProbabilities[1]);
        results.addModelResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p13, EMPTY_STRING, annotatedProbabilities[2]);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        attributeComputer(probabilities, score, probability);
        BOOST_TEST_REQUIRE(results.root());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(score, results.root()->s_RawAnomalyScore, 1e-12);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(probability, results.root()->probability(), 1e-12);
    }

    // Test over.
    {
        TDoubleVec const probabilities{0.25, 0.3, 0.001};
        TAnnotatedProbabilityVec annotatedProbabilities;
        for (auto p : probabilities) {
            annotatedProbabilities.emplace_back(p);
        }

        model::CHierarchicalResults results;
        results.addModelResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p11, EMPTY_STRING, annotatedProbabilities[0]);
        results.addModelResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p12, EMPTY_STRING, annotatedProbabilities[1]);
        results.addModelResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p13, EMPTY_STRING, annotatedProbabilities[2]);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nover:\n" << printer.result());
        personComputer(probabilities, score, probability);
        BOOST_TEST_REQUIRE(results.root());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(score, results.root()->s_RawAnomalyScore, 1e-12);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(probability, results.root()->probability(), 1e-12);
    }

    // Test aggregation of multiple searches.
    {
        std::array p11_ = {0.25, 0.3, 0.001};
        std::array p12_ = {0.2, 0.1};
        std::array p21_ = {0.5, 0.3};
        std::array p22_ = {0.025, 0.03};
        std::array rp1 = {0.006079029, 0.379477};
        std::array rp2 = {0.25, 0.001};
        std::array rp3 = {0.2, 0.1};
        model::SAnnotatedProbability annotatedProbability;

        model::CHierarchicalResults results;
        annotatedProbability.s_Probability = p11_[0];
        results.addModelResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p11, EMPTY_STRING, annotatedProbability);
        annotatedProbability.s_Probability = p11_[1];
        results.addModelResult(2, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p11, EMPTY_STRING, annotatedProbability);
        annotatedProbability.s_Probability = p11_[2];
        results.addModelResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p11, EMPTY_STRING, annotatedProbability);
        annotatedProbability.s_Probability = p12_[0];
        results.addModelResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p12, EMPTY_STRING, annotatedProbability);
        annotatedProbability.s_Probability = p12_[1];
        results.addModelResult(1, true, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p12, EMPTY_STRING, annotatedProbability);
        annotatedProbability.s_Probability = p21_[0];
        results.addModelResult(3, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF2, p21, EMPTY_STRING, annotatedProbability);
        annotatedProbability.s_Probability = p21_[1];
        results.addModelResult(3, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF2, p21, EMPTY_STRING, annotatedProbability);
        annotatedProbability.s_Probability = p22_[0];
        results.addModelResult(3, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF2, p22, EMPTY_STRING, annotatedProbability);
        annotatedProbability.s_Probability = p22_[1];
        results.addModelResult(3, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF2, p22, EMPTY_STRING, annotatedProbability);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\naggregates:\n" << printer.result());
        CNodeExtractor extract;
        results.bottomUpBreadthFirst(extract);
        TDoubleVec scores;
        TDoubleVec probabilities;
        for (const auto* i : extract.personNodes()) {
            scores.push_back(i->s_RawAnomalyScore);
            probabilities.push_back(i->probability());
        }
        maths::common::COrderings::simultaneousSort(probabilities, scores);
        TDoubleVec expectedScores;
        TDoubleVec expectedProbabilities;
        addAggregateValues(0.5, 0.5, 5, std::begin(rp1), std::end(rp1),
                           expectedScores, expectedProbabilities);
        addAggregateValues(0.5, 0.5, 5, std::begin(rp2), std::end(rp2),
                           expectedScores, expectedProbabilities);
        addAggregateValues(0.5, 0.5, 5, std::begin(rp3), std::end(rp3),
                           expectedScores, expectedProbabilities);
        maths::common::COrderings::simultaneousSort(expectedProbabilities, expectedScores);
        LOG_DEBUG(<< "expectedScores = " << expectedScores);
        LOG_DEBUG(<< "scores         = " << scores);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedScores),
                            core::CContainerPrinter::print(scores));
        LOG_DEBUG(<< "expectedProbabilities = " << expectedProbabilities);
        LOG_DEBUG(<< "probabilities         = " << probabilities);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedProbabilities),
                            core::CContainerPrinter::print(probabilities));
    }

    // Test partition
    {
        TDoubleVec const probabilities{0.01, 0.03, 0.001};
        TAnnotatedProbabilityVec annotatedProbabilities;
        for (auto p : probabilities) {
            annotatedProbabilities.emplace_back(p);
        }
        model::CHierarchicalResults results;
        results.addModelResult(1, false, FUNC, function, PNF1, pn11, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, annotatedProbabilities[0]);
        results.addModelResult(1, false, FUNC, function, PNF1, pn12, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, annotatedProbabilities[1]);
        results.addModelResult(1, false, FUNC, function, PNF1, pn13, EMPTY_STRING,
                               EMPTY_STRING, EMPTY_STRING, annotatedProbabilities[2]);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\npartition:\n" << printer.result());
        partitionComputer(probabilities, score, probability);
        BOOST_TEST_REQUIRE(results.root());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(score, results.root()->s_RawAnomalyScore, 1e-12);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(probability, results.root()->probability(), 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(testInfluence) {
    model::CAnomalyDetectorModelConfig const modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    model::CHierarchicalResultsAggregator aggregator(modelConfig);
    std::string const FUNC("max");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMax);

    std::string i2("i2");
    std::string i1("i1");
    std::string I("I");

    // Test by.
    {
        model::SAnnotatedProbability annotatedProbability1(0.22);
        annotatedProbability1.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 0.6);
        model::SAnnotatedProbability annotatedProbability2(0.003);
        annotatedProbability2.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 0.9);
        annotatedProbability2.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i2), 1.0);
        model::SAnnotatedProbability annotatedProbability3(0.01);
        annotatedProbability3.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 1.0);

        model::CHierarchicalResults results;
        results.addModelResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p11, EMPTY_STRING, annotatedProbability1);
        results.addModelResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p12, EMPTY_STRING, annotatedProbability2);
        results.addModelResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF1, p13, EMPTY_STRING, annotatedProbability3);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.createPivots();
        results.pivotsBottomUpBreadthFirst(aggregator);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        results.pivotsBottomUpBreadthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        BOOST_REQUIRE_EQUAL(
            std::string("'false/false////I//': 0.003600205, 0.02066228 pivot\n"
                        "  'false/false////I/i2/': 0.003, 0.0251169 pivot\n"
                        "  'false/false////I/i1/': 0.001801726, 0.04288765 pivot\n"
                        "'false/false////PF1//': 0.000885378, 0.08893476\n"
                        "  'false/false/max///PF1/p13/': 0.01, 0.008016032, [((I, i1), 1)]\n"
                        "  'false/false/max///PF1/p12/': 0.003, 0.03139613, [((I, i1), 0.9), ((I, i2), 1)]\n"
                        "  'false/false/max///PF1/p11/': 0.22, 0, [((I, i1), 0.6)]"),
            printer.result());
    }

    // Test complex.
    {
        model::SAnnotatedProbability annotatedProbability1(0.22);
        annotatedProbability1.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 0.6);
        model::SAnnotatedProbability annotatedProbability2(0.003);
        annotatedProbability2.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 0.9);
        annotatedProbability2.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i2), 1.0);
        model::SAnnotatedProbability annotatedProbability3(0.01);
        annotatedProbability3.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 1.0);
        model::SAnnotatedProbability annotatedProbability4(0.03);
        annotatedProbability4.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 0.6);
        annotatedProbability4.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i2), 0.8);
        model::SAnnotatedProbability annotatedProbability5(0.56);
        annotatedProbability5.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 0.8);

        model::CHierarchicalResults results;
        results.addModelResult(1, true, FUNC, function, PNF1, pn11, PF1, p11,
                               EMPTY_STRING, annotatedProbability1);
        results.addModelResult(1, true, FUNC, function, PNF1, pn12, PF1, p12,
                               EMPTY_STRING, annotatedProbability2);
        results.addModelResult(2, false, FUNC, function, PNF2, pn21, PF1, p13,
                               EMPTY_STRING, annotatedProbability3);
        results.addModelResult(2, false, FUNC, function, PNF2, pn22, PF1, p12,
                               EMPTY_STRING, annotatedProbability4);
        results.addModelResult(2, false, FUNC, function, PNF2, pn23, PF1, p12,
                               EMPTY_STRING, annotatedProbability5);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.createPivots();
        results.pivotsBottomUpBreadthFirst(aggregator);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        results.pivotsBottomUpBreadthFirst(printer);
        LOG_DEBUG(<< "\ncomplex:\n" << printer.result());
        BOOST_REQUIRE_EQUAL(
            std::string("'false/false////I//': 0.006210884, 0.01130322 pivot\n"
                        "  'false/false////I/i2/': 0.003110279, 0.0241695 pivot\n"
                        "  'false/false////I/i1/': 0.00619034, 0.01134605 pivot\n"
                        "'false/true//////': 0.003651953, 0.02034678\n"
                        "  'false/false//PNF2////': 0.029701, 0.001095703\n"
                        "    'false/false/max/PNF2/pn23/PF1/p12/': 0.56, 0, [((I, i1), 0.8)]\n"
                        "    'false/false/max/PNF2/pn22/PF1/p12/': 0.03, 0.001336005, [((I, i1), 0.6), ((I, i2), 0.8)]\n"
                        "    'false/false/max/PNF2/pn21/PF1/p13/': 0.01, 0.008016032, [((I, i1), 1)]\n"
                        "  'false/true//PNF1////': 0.005991, 0.01177692\n"
                        "    'false/true/max/PNF1/pn12/PF1/p12/': 0.003, 0.03139613, [((I, i1), 0.9), ((I, i2), 1)]\n"
                        "    'false/true/max/PNF1/pn11/PF1/p11/': 0.22, 0, [((I, i1), 0.6)]"),
            printer.result());
    }

    // Test high probability records are written due to low probability influencer
    {
        model::SAnnotatedProbability annotatedProbability1Low(0.06);
        annotatedProbability1Low.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i1), 1.0);
        model::SAnnotatedProbability const annotatedProbability1High(0.8);
        model::SAnnotatedProbability annotatedProbability11 = annotatedProbability1Low;
        model::SAnnotatedProbability annotatedProbability12 = annotatedProbability1High;
        model::SAnnotatedProbability annotatedProbability13 = annotatedProbability1Low;
        model::SAnnotatedProbability annotatedProbability14 = annotatedProbability1High;
        model::SAnnotatedProbability annotatedProbability15 = annotatedProbability1High;
        model::SAnnotatedProbability annotatedProbability16 = annotatedProbability1High;
        model::SAnnotatedProbability annotatedProbability2(0.001);
        annotatedProbability2.s_Influences.emplace_back(
            TOptionalStrOptionalStrPr(I, i2), 1.0);

        model::CHierarchicalResults results;
        results.addInfluencer(I);
        results.addModelResult(1, false, FUNC, function, PNF1, pn11, PF1, p11,
                               EMPTY_STRING, annotatedProbability11);
        results.addModelResult(1, false, FUNC, function, PNF1, pn11, PF1, p12,
                               EMPTY_STRING, annotatedProbability12);
        results.addModelResult(1, false, FUNC, function, PNF1, pn11, PF1, p13,
                               EMPTY_STRING, annotatedProbability13);
        results.addModelResult(1, false, FUNC, function, PNF1, pn11, PF1, p14,
                               EMPTY_STRING, annotatedProbability14);
        results.addModelResult(1, false, FUNC, function, PNF1, pn11, PF1, p15,
                               EMPTY_STRING, annotatedProbability15);
        results.addModelResult(1, false, FUNC, function, PNF1, pn11, PF1, p16,
                               EMPTY_STRING, annotatedProbability16);
        results.addModelResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                               PF2, p21, EMPTY_STRING, annotatedProbability2);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.createPivots();
        results.pivotsBottomUpBreadthFirst(aggregator);

        CPrinter writtenNodesOnlyPrinter(true);
        results.postorderDepthFirst(writtenNodesOnlyPrinter);
        results.pivotsBottomUpBreadthFirst(writtenNodesOnlyPrinter);
        LOG_DEBUG(<< "\nhigh p records with low p influencer:\n"
                  << writtenNodesOnlyPrinter.result());
        BOOST_REQUIRE_EQUAL(
            std::string("'false/false////I//': 0.001999, 0.038497 pivot\n"
                        "  'false/false////I/i2/': 0.001, 0.07855711 pivot\n"
                        "  'false/false////I/i1/': 0.01939367, 0.002530117 pivot\n"
                        "'false/false//////': 0.001999, 0.038497\n"
                        "    'false/false/max/PNF1/pn11/PF1/p13/': 0.06, 0, [((I, i1), 1)]\n"
                        "    'false/false/max/PNF1/pn11/PF1/p11/': 0.06, 0, [((I, i1), 1)]\n"
                        "  'false/false/max///PF2/p21/': 0.001, 0.09819639, [((I, i2), 1)]"),
            writtenNodesOnlyPrinter.result());
    }
}

BOOST_AUTO_TEST_CASE(testScores) {
    model::CAnomalyDetectorModelConfig const modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    model::CLimits const limits;
    model::CHierarchicalResultsAggregator aggregator(modelConfig);
    model::CHierarchicalResultsProbabilityFinalizer finalizer;
    CCheckScores checkScores;
    static const std::string MAX("max");
    static const std::string RARE("rare");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMax);

    // Test vanilla by / over.
    {
        model::CHierarchicalResults results;
        addResult(1, false, MAX, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 1.0, results);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.bottomUpBreadthFirst(finalizer);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        results.bottomUpBreadthFirst(checkScores);
    }
    {
        model::CHierarchicalResults results;
        addResult(1, false, MAX, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.6, results);
        addResult(1, false, MAX, function, EMPTY_STRING, EMPTY_STRING, PF1, p12,
                  EMPTY_STRING, 0.7, results);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.bottomUpBreadthFirst(finalizer);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nby:\n" << printer.result());
        results.bottomUpBreadthFirst(checkScores);
    }
    {
        model::CHierarchicalResults results;
        addResult(1, false, MAX, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 0.3, results);
        addResult(2, true, MAX, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.01, results);
        addResult(2, true, MAX, function, EMPTY_STRING, EMPTY_STRING, PF1, p12,
                  EMPTY_STRING, 0.03, results);
        addResult(3, false, MAX, function, EMPTY_STRING, EMPTY_STRING, PF2, p22,
                  EMPTY_STRING, 0.03, results);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.bottomUpBreadthFirst(finalizer);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nover:\n" << printer.result());
        results.bottomUpBreadthFirst(checkScores);
    }
    {
        model::CHierarchicalResults results;
        addResult(1, true, MAX, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.01, results);
        addResult(1, true, MAX, function, EMPTY_STRING, EMPTY_STRING, PF1, p12,
                  EMPTY_STRING, 0.03, results);
        addResult(2, true, RARE, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.07, results);
        addResult(2, true, RARE, function, EMPTY_STRING, EMPTY_STRING, PF1, p12,
                  EMPTY_STRING, 0.3, results);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.bottomUpBreadthFirst(finalizer);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\nover:\n" << printer.result());
        results.bottomUpBreadthFirst(checkScores);
    }

    // Test vanilla partition
    {
        model::CHierarchicalResults results;
        addResult(1, false, MAX, function, PNF1, pn11, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(1, false, MAX, function, PNF1, pn12, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(1, false, MAX, function, PNF1, pn13, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.05, results);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.bottomUpBreadthFirst(finalizer);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\npartition\n:" << printer.result());
        results.bottomUpBreadthFirst(checkScores);
    }

    // Test complex.
    {
        model::CHierarchicalResults results;
        addResult(1, false, MAX, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(2, false, MAX, function, EMPTY_STRING, EMPTY_STRING, PF1, p11,
                  EMPTY_STRING, 0.01, results);
        addResult(2, false, MAX, function, EMPTY_STRING, EMPTY_STRING, PF1, p14,
                  EMPTY_STRING, 0.01, results);
        addResult(3, false, MAX, function, PNF1, pn11, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(3, false, MAX, function, PNF1, pn12, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(3, false, MAX, function, PNF1, pn13, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.05, results);
        addResult(4, true, MAX, function, PNF2, pn22, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.01, results);
        addResult(4, true, MAX, function, PNF2, pn23, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.05, results);
        addResult(5, true, MAX, function, PNF2, pn21, PF1, p11, EMPTY_STRING, 0.2, results);
        addResult(5, true, MAX, function, PNF2, pn22, PF1, p11, EMPTY_STRING, 0.2, results);
        addResult(5, true, MAX, function, PNF2, pn22, PF1, p12, EMPTY_STRING, 0.1, results);
        addResult(6, true, MAX, function, PNF2, pn22, PF2, p21, EMPTY_STRING, 0.15, results);
        addResult(7, false, MAX, function, PNF2, pn22, PF2, p21, EMPTY_STRING, 0.12, results);
        addResult(6, true, MAX, function, PNF2, pn22, PF2, p23, EMPTY_STRING, 0.12, results);
        addResult(7, false, MAX, function, PNF2, pn22, PF2, p23, EMPTY_STRING, 0.82, results);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.bottomUpBreadthFirst(finalizer);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\ncomplex:\n" << printer.result());
        results.bottomUpBreadthFirst(checkScores);
    }
}

BOOST_AUTO_TEST_CASE(testWriter) {
    model::CAnomalyDetectorModelConfig const modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    model::CLimits const limits;
    model::CHierarchicalResultsAggregator aggregator(modelConfig);
    CWriteConsistencyChecker writeConsistencyChecker(limits);

    static const std::string FUNC("max");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMax);

    // Test complex.
    {
        model::CResourceMonitor resourceMonitor;
        using TStrCPtrVec = model::CDataGatherer::TStrCPtrVec;
        model::SModelParams const params(modelConfig.bucketLength());
        auto interimBucketCorrector =
            std::make_shared<model::CInterimBucketCorrector>(modelConfig.bucketLength());
        model::CSearchKey key;
        auto dataGatherer =
            model::CDataGathererBuilder(model_t::E_EventRate,
                                        {model_t::E_IndividualCountByBucketAndPerson},
                                        params, key, modelConfig.bucketLength())
                .buildSharedPtr();
        model::CEventData dummy;
        dataGatherer->addArrival(TStrCPtrVec(1, &EMPTY_STRING), dummy, resourceMonitor);
        dummy.clear();
        dataGatherer->addArrival(TStrCPtrVec(1, &p11), dummy, resourceMonitor);
        dummy.clear();
        dataGatherer->addArrival(TStrCPtrVec(1, &p12), dummy, resourceMonitor);
        dummy.clear();
        dataGatherer->addArrival(TStrCPtrVec(1, &p14), dummy, resourceMonitor);
        dummy.clear();
        dataGatherer->addArrival(TStrCPtrVec(1, &p21), dummy, resourceMonitor);
        dummy.clear();
        dataGatherer->addArrival(TStrCPtrVec(1, &p23), dummy, resourceMonitor);
        model::CCountingModel model(params, dataGatherer, interimBucketCorrector);
        model::CHierarchicalResults results;
        addResult(1, false, FUNC, function, EMPTY_STRING, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, 0.001, &model, results);
        addResult(2, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  p11, EMPTY_STRING, 0.001, &model, results);
        addResult(2, false, FUNC, function, EMPTY_STRING, EMPTY_STRING, PF1,
                  p14, EMPTY_STRING, 0.001, &model, results);
        addResult(3, false, FUNC, function, PNF1, pn11, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.001, &model, results);
        addResult(3, false, FUNC, function, PNF1, pn12, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.001, &model, results);
        addResult(3, false, FUNC, function, PNF1, pn13, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.005, &model, results);
        addResult(4, true, FUNC, function, PNF2, pn22, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.001, &model, results);
        addResult(4, true, FUNC, function, PNF2, pn23, EMPTY_STRING,
                  EMPTY_STRING, EMPTY_STRING, 0.005, &model, results);
        addResult(5, true, FUNC, function, PNF2, pn21, PF1, p11, EMPTY_STRING,
                  0.008, &model, results);
        addResult(5, true, FUNC, function, PNF2, pn22, PF1, p11, EMPTY_STRING,
                  0.009, &model, results);
        addResult(5, true, FUNC, function, PNF2, pn22, PF1, p12, EMPTY_STRING,
                  0.01, &model, results);
        addResult(6, true, FUNC, function, PNF2, pn22, PF2, p21, EMPTY_STRING,
                  0.007, &model, results);
        addResult(7, false, FUNC, function, PNF2, pn22, PF2, p21, EMPTY_STRING,
                  0.006, &model, results);
        addResult(6, true, FUNC, function, PNF2, pn22, PF2, p23, EMPTY_STRING,
                  0.004, &model, results);
        addResult(7, false, FUNC, function, PNF2, pn22, PF2, p23, EMPTY_STRING,
                  0.003, &model, results);
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        CPrinter printer;
        results.postorderDepthFirst(printer);
        LOG_DEBUG(<< "\ncomplex:\n" << printer.result());
        results.bottomUpBreadthFirst(writeConsistencyChecker);
    }
}

BOOST_AUTO_TEST_CASE(testNormalizer) {
    using TNormalizerPtr = std::shared_ptr<model::CAnomalyScore::CNormalizer>;
    using TStrNormalizerPtrMap = std::map<std::string, TNormalizerPtr>;
    using TNodeCPtrSet = std::set<const model::CHierarchicalResultsVisitor::TNode*>;

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    model::CHierarchicalResultsAggregator aggregator(modelConfig);
    model::CHierarchicalResultsProbabilityFinalizer finalizer;
    model::CLimits l;
    model::CHierarchicalResultsNormalizer normalizer(l, modelConfig);
    static const std::string FUNC("max");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMax);

    // Not using TRUE and FALSE as they clash with Windows macros

    const std::array<std::array<std::string, 7>, 9> fields = {
        {{"1", FALSE_STR, PNF1, pn11, PF2, p21, EMPTY_STRING},
         {"1", FALSE_STR, PNF1, pn11, PF2, p22, EMPTY_STRING},
         {"1", FALSE_STR, PNF1, pn11, PF2, p23, EMPTY_STRING},
         {"2", TRUE_STR, PNF1, pn12, PF1, p11, EMPTY_STRING},
         {"2", TRUE_STR, PNF1, pn12, PF1, p12, EMPTY_STRING},
         {"2", TRUE_STR, PNF1, pn12, PF1, p13, EMPTY_STRING},
         {"3", FALSE_STR, PNF2, pn21, PF1, p11, EMPTY_STRING},
         {"3", FALSE_STR, PNF2, pn22, PF1, p12, EMPTY_STRING},
         {"3", FALSE_STR, PNF2, pn23, PF1, p13, EMPTY_STRING}}};
    TStrNormalizerPtrMap expectedNormalizers;
    expectedNormalizers.emplace(
        "r", std::make_shared<model::CAnomalyScore::CNormalizer>(modelConfig));
    test::CRandomNumbers rng;

    for (std::size_t i = 0; i < 300; ++i) {
        model::CHierarchicalResults results;
        TDoubleVec p;
        rng.generateUniformSamples(0.0, 1.0, std::size(fields), p);
        const TAttributeProbabilityVec empty;
        for (std::size_t j = 0; j < std::size(fields); ++j) {
            addResult(boost::lexical_cast<int>(fields[j][0]), fields[j][1] == TRUE_STR,
                      FUNC, function, fields[j][2], fields[j][3], fields[j][4],
                      fields[j][5], fields[j][6], p[j], results);
        }
        results.buildHierarchy();
        results.bottomUpBreadthFirst(aggregator);
        results.bottomUpBreadthFirst(finalizer);
        normalizer.setJob(model::CHierarchicalResultsNormalizer::E_RefreshSettings);
        results.bottomUpBreadthFirst(normalizer);
        normalizer.setJob(model::CHierarchicalResultsNormalizer::E_UpdateQuantiles);
        results.bottomUpBreadthFirst(normalizer);
        normalizer.setJob(model::CHierarchicalResultsNormalizer::E_NormalizeScores);
        results.bottomUpBreadthFirst(normalizer);

        CNodeExtractor extract;
        results.bottomUpBreadthFirst(extract);

        LOG_TRACE(<< "** Iteration = " << i << " **");
        TNodeCPtrSet nodes;

        TDoubleVec normalized;
        TDoubleVec expectedNormalized;

        auto findOrInsertExpectedNormalizer =
            [&expectedNormalizers, &modelConfig](
                std::string key, const model::CHierarchicalResultsVisitor::TNode* node) {
                const std::string& partitionFieldName = *node->s_Spec.s_PartitionFieldName;
                const std::string& personFieldName =
                    key == "n" ? EMPTY_STRING : *node->s_Spec.s_PersonFieldName;
                key += ' ' + partitionFieldName;
                key += ' ' + personFieldName;
                auto entry = expectedNormalizers.find(key);
                if (entry != expectedNormalizers.end()) {
                    return entry->second;
                }
                TNormalizerPtr& result = expectedNormalizers[key];
                result = std::make_shared<model::CAnomalyScore::CNormalizer>(modelConfig);
                result->isForMembersOfPopulation((partitionFieldName == PNF1) &&
                                                 (personFieldName == PF1));
                return result;
            };
        auto scope = [](const model::CHierarchicalResultsVisitor::TNode* node) {
            return model::CAnomalyScore::CNormalizer::CMaximumScoreScope{
                node->s_Spec.s_PartitionFieldName, node->s_Spec.s_PartitionFieldValue,
                node->s_Spec.s_PersonFieldName, node->s_Spec.s_PersonFieldValue};
        };

        for (const auto& leaf : extract.leafNodes()) {
            auto expectedNormalizer = findOrInsertExpectedNormalizer("l", leaf);
            double const probability = leaf->probability();
            // This truncation condition needs to be kept the same as the one in
            // CHierarchicalResultsNormalizer::visit()
            double const score = probability > modelConfig.maximumAnomalousProbability()
                                     ? 0.0
                                     : maths::common::CTools::anomalyScore(probability);
            expectedNormalizer->updateQuantiles(scope(leaf), score);
        }
        for (const auto& leaf : extract.leafNodes()) {
            auto expectedNormalizer = findOrInsertExpectedNormalizer("l", leaf);
            if (nodes.insert(leaf).second) {
                double const probability = leaf->probability();
                // This truncation condition needs to be kept the same as the one in
                // CHierarchicalResultsNormalizer::visit()
                double score = probability > modelConfig.maximumAnomalousProbability()
                                   ? 0.0
                                   : maths::common::CTools::anomalyScore(probability);
                normalized.push_back(leaf->s_NormalizedAnomalyScore);
                BOOST_TEST_REQUIRE(expectedNormalizer->normalize(scope(leaf), score));
                expectedNormalized.push_back(score);
            }
        }
        LOG_TRACE(<< "* leaf *");
        LOG_TRACE(<< "expectedNormalized = " << expectedNormalized);
        LOG_TRACE(<< "normalized         = " << normalized);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedNormalized),
                            core::CContainerPrinter::print(normalized));

        normalized.clear();
        expectedNormalized.clear();
        for (const auto& person : extract.personNodes()) {
            auto expectedNormalizer = findOrInsertExpectedNormalizer("p", person);
            double const probability = person->probability();
            // This truncation condition needs to be kept the same as the one in
            // CHierarchicalResultsNormalizer::visit()
            double const score = probability > modelConfig.maximumAnomalousProbability()
                                     ? 0.0
                                     : maths::common::CTools::anomalyScore(probability);
            expectedNormalizer->updateQuantiles(scope(person), score);
        }
        for (const auto& person : extract.personNodes()) {
            auto expectedNormalizer = findOrInsertExpectedNormalizer("p", person);
            if (nodes.insert(person).second) {
                double const probability = person->probability();
                // This truncation condition needs to be kept the same as the one in
                // CHierarchicalResultsNormalizer::visit()
                double score = probability > modelConfig.maximumAnomalousProbability()
                                   ? 0.0
                                   : maths::common::CTools::anomalyScore(probability);
                normalized.push_back(person->s_NormalizedAnomalyScore);
                BOOST_TEST_REQUIRE(expectedNormalizer->normalize(scope(person), score));
                expectedNormalized.push_back(score);
            }
        }
        LOG_TRACE(<< "* person *");
        LOG_TRACE(<< "expectedNormalized = " << expectedNormalized);
        LOG_TRACE(<< "normalized         = " << normalized);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedNormalized),
                            core::CContainerPrinter::print(normalized));

        normalized.clear();
        expectedNormalized.clear();
        for (const auto& partition : extract.partitionNodes()) {
            auto expectedNormalizer = findOrInsertExpectedNormalizer("n", partition);
            double const probability = partition->probability();
            // This truncation condition needs to be kept the same as the one in
            // CHierarchicalResultsNormalizer::visit()
            double const score = probability > modelConfig.maximumAnomalousProbability()
                                     ? 0.0
                                     : maths::common::CTools::anomalyScore(probability);
            expectedNormalizer->updateQuantiles(scope(partition), score);
        }
        for (const auto& partition : extract.partitionNodes()) {
            auto expectedNormalizer = findOrInsertExpectedNormalizer("n", partition);
            if (nodes.insert(partition).second) {
                double const probability = partition->probability();
                // This truncation condition needs to be kept the same as the one in
                // CHierarchicalResultsNormalizer::visit()
                double score = probability > modelConfig.maximumAnomalousProbability()
                                   ? 0.0
                                   : maths::common::CTools::anomalyScore(probability);
                normalized.push_back(partition->s_NormalizedAnomalyScore);
                BOOST_TEST_REQUIRE(expectedNormalizer->normalize(scope(partition), score));
                expectedNormalized.push_back(score);
            }
        }
        LOG_TRACE(<< "* partition *");
        LOG_TRACE(<< "expectedNormalized = " << expectedNormalized);
        LOG_TRACE(<< "normalized         = " << normalized);
        BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedNormalized),
                            core::CContainerPrinter::print(normalized));

        double const probability = results.root()->probability();
        // This truncation condition needs to be kept the same as the one in
        // CHierarchicalResultsNormalizer::visit()
        double score = probability > modelConfig.maximumAnomalousProbability()
                           ? 0.0
                           : maths::common::CTools::anomalyScore(probability);

        TOptionalStr personFieldName = "bucket_time";
        expectedNormalizers.find(std::string("r"))->second->isForMembersOfPopulation(false);
        expectedNormalizers.find(std::string("r"))
            ->second->updateQuantiles({EMPTY_OPTIONAL_STR, EMPTY_OPTIONAL_STR,
                                       EMPTY_OPTIONAL_STR, EMPTY_OPTIONAL_STR},
                                      score);
        expectedNormalizers.find(std::string("r"))
            ->second->normalize({EMPTY_OPTIONAL_STR, EMPTY_OPTIONAL_STR,
                                 personFieldName, EMPTY_OPTIONAL_STR},
                                score);
        LOG_TRACE(<< "* root *");
        LOG_TRACE(<< "expectedNormalized = " << results.root()->s_NormalizedAnomalyScore);
        LOG_TRACE(<< "normalized         = " << score);
        BOOST_REQUIRE_CLOSE_ABSOLUTE(results.root()->s_NormalizedAnomalyScore, score, 1e-10);
    }

    // Test JSON round-tripping
    std::string origJson;
    normalizer.toJson(123, "mykey", origJson, true);
    BOOST_TEST_REQUIRE(!origJson.empty());
    LOG_DEBUG(<< "Compressed JSON doc is:\n" << origJson);

    {
        model::CLimits limits;
        model::CHierarchicalResultsNormalizer newNormalizerJson(limits, modelConfig);
        std::stringstream stream(origJson);
        BOOST_REQUIRE_EQUAL(model::CHierarchicalResultsNormalizer::E_Ok,
                            newNormalizerJson.fromJsonStream(stream));

        std::string newJson;
        newNormalizerJson.toJson(123, "mykey", newJson, true);
        BOOST_REQUIRE_EQUAL(newJson, origJson);
    }

    // Test it still works if what we restore was uncompressed.
    // (This will be the case for quantiles persisted by old versions.)
    {
        std::string uncompressedJson;
        std::istringstream streamToDecompress{origJson};
        using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
        TFilteredInput filteredInput;
        filteredInput.push(boost::iostreams::gzip_decompressor());
        filteredInput.push(core::CBase64Decoder{});
        filteredInput.push(streamToDecompress);
        std::array<char, 100> buf;
        do {
            filteredInput.read(buf.data(), buf.size());
            std::streamsize const num{filteredInput.gcount()};
            if (filteredInput.bad() == false && num > 0) {
                uncompressedJson.append(buf.data(), num);
            }
        } while (filteredInput);
        LOG_DEBUG(<< "Uncompressed JSON doc is:\n" << uncompressedJson);

        model::CLimits limits;
        model::CHierarchicalResultsNormalizer newNormalizerJson(limits, modelConfig);
        std::stringstream stream(uncompressedJson);
        BOOST_REQUIRE_EQUAL(model::CHierarchicalResultsNormalizer::E_Ok,
                            newNormalizerJson.fromJsonStream(stream));

        std::string newJson;
        newNormalizerJson.toJson(123, "mykey", newJson, true);
        BOOST_REQUIRE_EQUAL(newJson, origJson);
    }
}

BOOST_AUTO_TEST_CASE(testDetectorEqualizing) {
    model::CAnomalyDetectorModelConfig const modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    test::CRandomNumbers rng;

    {
        model::CHierarchicalResultsAggregator aggregator(modelConfig);
        aggregator.setJob(model::CHierarchicalResultsAggregator::E_UpdateAndCorrect);
        CProbabilityGatherer probabilityGatherer;
        static const std::string FUNC("max");
        static constexpr ml::model::function_t::EFunction function(
            ml::model::function_t::E_IndividualMetricMax);

        const std::array<std::array<std::string, 7>, 12> fields = {
            {{"0", FALSE_STR, PNF1, pn11, PF1, p11, EMPTY_STRING},
             {"0", FALSE_STR, PNF1, pn12, PF1, p12, EMPTY_STRING},
             {"0", FALSE_STR, PNF1, pn11, PF1, p12, EMPTY_STRING},
             {"1", FALSE_STR, PNF1, pn11, PF1, p11, EMPTY_STRING},
             {"1", FALSE_STR, PNF1, pn12, PF1, p12, EMPTY_STRING},
             {"1", FALSE_STR, PNF1, pn11, PF1, p12, EMPTY_STRING},
             {"2", TRUE_STR, PNF1, pn12, PF1, p11, EMPTY_STRING},
             {"2", TRUE_STR, PNF1, pn12, PF1, p12, EMPTY_STRING},
             {"2", TRUE_STR, PNF1, pn11, PF1, p12, EMPTY_STRING},
             {"3", FALSE_STR, PNF1, pn11, PF1, p11, EMPTY_STRING},
             {"3", FALSE_STR, PNF1, pn12, PF1, p12, EMPTY_STRING},
             {"3", FALSE_STR, PNF1, pn12, PF1, p12, EMPTY_STRING}}};
        constexpr std::array scales = {1.9, 2.5, 1.7, 2.9};

        for (std::size_t i = 0; i < 300; ++i) {
            model::CHierarchicalResults results;
            const TAttributeProbabilityVec empty;
            for (const auto& field : fields) {
                int const detector = boost::lexical_cast<int>(field[0]);
                TDoubleVec p;
                rng.generateGammaSamples(1.0, scales[detector], 1, p);
                p[0] = std::exp(-p[0]);
                addResult(detector, field[1] == TRUE_STR, FUNC, function, field[2],
                          field[3], field[4], field[5], field[6], p[0], results);
            }
            results.buildHierarchy();
            results.bottomUpBreadthFirst(aggregator);
        }

        for (std::size_t i = 0; i < 300; ++i) {
            model::CHierarchicalResults results;
            const TAttributeProbabilityVec empty;
            for (const auto& field : fields) {
                int const detector = boost::lexical_cast<int>(field[0]);
                TDoubleVec p;
                rng.generateGammaSamples(1.0, scales[detector], 1, p);
                p[0] = std::exp(-p[0]);
                addResult(detector, field[1] == TRUE_STR, FUNC, function, field[2],
                          field[3], field[4], field[5], field[6], p[0], results);
            }
            results.buildHierarchy();
            results.bottomUpBreadthFirst(aggregator);
            results.bottomUpBreadthFirst(probabilityGatherer);
        }

        double significance = probabilityGatherer.test(8e-7);
        LOG_DEBUG(<< "total significance = " << significance);

        BOOST_TEST_REQUIRE(significance > 0.002);

        std::ostringstream origJson;
        core::CJsonStatePersistInserter::persist(
            origJson, std::bind_front(&model::CHierarchicalResultsAggregator::acceptPersistInserter,
                                      &aggregator));
        LOG_DEBUG(<< "aggregator JSON representation:\n" << origJson.str());

        model::CHierarchicalResultsAggregator restoredAggregator(modelConfig);
        {
            // The traverser expects the state json in a embedded document
            std::istringstream origJsonStrm("{\"topLevel\" : " + origJson.str() + "}");
            core::CJsonStateRestoreTraverser traverser(origJsonStrm);
            BOOST_TEST_REQUIRE(traverser.traverseSubLevel(std::bind_front(
                &model::CHierarchicalResultsAggregator::acceptRestoreTraverser,
                &restoredAggregator)));
        }

        // Checksums should agree.
        BOOST_REQUIRE_EQUAL(aggregator.checksum(), restoredAggregator.checksum());

        // The persist and restore should be idempotent.
        std::ostringstream newJson;
        core::CJsonStatePersistInserter::persist(
            newJson, std::bind_front(&model::CHierarchicalResultsAggregator::acceptPersistInserter,
                                     &restoredAggregator));
        BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
    }
    {
        model::CHierarchicalResultsAggregator aggregator(modelConfig);
        aggregator.setJob(model::CHierarchicalResultsAggregator::E_UpdateAndCorrect);
        static const std::string FUNC("max");
        static constexpr ml::model::function_t::EFunction function(
            ml::model::function_t::E_IndividualMetricMax);

        const std::array<std::array<std::string, 7>, 2> fields = {
            {{"0", FALSE_STR, PNF1, pn11, PF1, p11, EMPTY_STRING},
             {"1", FALSE_STR, PNF1, pn11, PF1, p11, EMPTY_STRING}}};
        constexpr std::array scales = {1.0, 3.5};

        for (std::size_t i = 0; i < 500; ++i) {
            model::CHierarchicalResults results;
            const TAttributeProbabilityVec empty;
            for (const auto& field : fields) {
                int const detector = boost::lexical_cast<int>(field[0]);
                TDoubleVec p;
                rng.generateGammaSamples(1.0, scales[detector], 1, p);
                p[0] = std::exp(-p[0]);
                addResult(detector, field[1] == TRUE_STR, FUNC, function, field[2],
                          field[3], field[4], field[5], field[6], p[0], results);
            }
            results.buildHierarchy();
            results.bottomUpBreadthFirst(aggregator);
        }

        using TDoubleSizePr = std::pair<double, std::size_t>;
        maths::common::CBasicStatistics::COrderStatisticsStack<TDoubleSizePr, 2> mostAnomalous;

        for (std::size_t i = 0; i < 100; ++i) {
            model::CHierarchicalResults results;
            const TAttributeProbabilityVec empty;
            for (const auto& field : fields) {
                int const detector = boost::lexical_cast<int>(field[0]);
                TDoubleVec p;
                rng.generateGammaSamples(1.0, scales[detector], 1, p);
                p[0] = detector == 0 && i == 70 ? 2.1e-5 : std::exp(-p[0]);
                addResult(detector, field[1] == TRUE_STR, FUNC, function, field[2],
                          field[3], field[4], field[5], field[6], p[0], results);
            }
            results.buildHierarchy();
            results.bottomUpBreadthFirst(aggregator);

            mostAnomalous.add(std::make_pair(
                results.root()->s_AnnotatedProbability.s_Probability, i));
        }

        mostAnomalous.sort();
        LOG_DEBUG(<< "mostAnomalousBucket = " << mostAnomalous);
        BOOST_REQUIRE_EQUAL(70, mostAnomalous[0].second);
        BOOST_TEST_REQUIRE(mostAnomalous[0].first / mostAnomalous[1].first < 100);
    }
}

BOOST_AUTO_TEST_CASE(testShouldWritePartition) {
    static const std::string PART1("PART1");
    static const std::string PERS("PERS");
    std::string const pers1("pers1");
    std::string const pers2("pers2");
    static const std::string VAL1("VAL1");
    std::string const partition1("par_1");
    std::string const partition2("par_2");

    static const std::string FUNC("mean");
    static constexpr ml::model::function_t::EFunction function(
        ml::model::function_t::E_IndividualMetricMean);

    model::CHierarchicalResults results;
    addResult(1, false, FUNC, function, PART1, partition1, PERS, pers1, VAL1, 0.001, results);
    addResult(1, false, FUNC, function, PART1, partition2, PERS, pers1, VAL1, 0.001, results);
    addResult(1, false, FUNC, function, PART1, partition2, PERS, pers2, VAL1, 0.001, results);

    results.buildHierarchy();
    CPrinter printer;
    results.postorderDepthFirst(printer);
    LOG_DEBUG(<< "\nhierarchy:\n" << printer.result());

    const ml::model::CHierarchicalResults::TNode* root = results.root();
    BOOST_REQUIRE_EQUAL(2, root->s_Children.size());

    CNodeExtractor extract;
    results.bottomUpBreadthFirst(extract);
    BOOST_REQUIRE_EQUAL(1, extract.partitionedNodes().size());
    BOOST_REQUIRE_EQUAL(2, extract.partitionNodes().size());

    BOOST_REQUIRE_EQUAL(2, extract.personNodes().size());
    BOOST_REQUIRE_EQUAL(3, extract.leafNodes().size());

    LOG_DEBUG(<< "Partition 1 child count "
              << extract.partitionNodes()[0]->s_Children.size());
    LOG_DEBUG(<< "Partition 2 child count "
              << extract.partitionNodes()[1]->s_Children.size());

    BOOST_REQUIRE_EQUAL(0, extract.partitionNodes()[0]->s_Children.size());
    BOOST_REQUIRE_EQUAL(2, extract.partitionNodes()[1]->s_Children.size());

    model::CAnomalyDetectorModelConfig const modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig();
    ml::model::CHierarchicalResultsAggregator aggregator(modelConfig);
    results.bottomUpBreadthFirst(aggregator);

    model::CLimits const limits;
    BOOST_TEST_REQUIRE(ml::model::CHierarchicalResultsVisitor::shouldWriteResult(
        limits, results, *extract.partitionNodes()[0], false));
    BOOST_TEST_REQUIRE(ml::model::CHierarchicalResultsVisitor::shouldWriteResult(
        limits, results, *extract.partitionNodes()[1], false));
}

BOOST_AUTO_TEST_SUITE_END()
