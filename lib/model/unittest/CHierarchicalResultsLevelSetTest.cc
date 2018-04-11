/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include "CHierarchicalResultsLevelSetTest.h"

#include <core/CStoredStringPtr.h>

#include <model/CAnnotatedProbability.h>
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsLevelSet.h>
#include <model/CStringStore.h>

CppUnit::Test* CHierarchicalResultsLevelSetTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CHierarchicalResultsLevelSetTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CHierarchicalResultsLevelSetTest>(
        "CHierarchicalResultsLevelSetTest::"
        "testElementsWithPerPartitionNormalisation",
        &CHierarchicalResultsLevelSetTest::testElementsWithPerPartitionNormalisation));

    return suiteOfTests;
}

struct TestNode {
    TestNode(const std::string& name) : s_Name(name) {}

    std::string s_Name;
};

class CTestNodeFactory {
public:
    CTestNodeFactory() {}

    TestNode make(const std::string& name1,
                  const std::string& name2,
                  const std::string& name3,
                  const std::string& name4) const {
        return make(name1 + ' ' + name2 + ' ' + name3 + ' ' + name4);
    }

    TestNode make(const std::string& name1, const std::string& name2) const {
        return make(name1 + ' ' + name2);
    }

    TestNode make(const std::string& name) const { return TestNode(name); }
};

class CConcreteHierarchicalResultsLevelSet
    : public ml::model::CHierarchicalResultsLevelSet<TestNode> {
public:
    CConcreteHierarchicalResultsLevelSet(const TestNode& root)
        : ml::model::CHierarchicalResultsLevelSet<TestNode>(root) {}

    //! Visit a node.
    virtual void visit(const ml::model::CHierarchicalResults& /*results*/,
                       const TNode& /*node*/,
                       bool /*pivot*/) {}

    // make public
    using ml::model::CHierarchicalResultsLevelSet<TestNode>::elements;
};

void print(const TestNode* node) {
    std::cout << "'" << node->s_Name << "'" << std::endl;
}

void CHierarchicalResultsLevelSetTest::testElementsWithPerPartitionNormalisation() {
    // This is intentionally NOT an empty string from the string store, but
    // instead a completely separate empty string, such that its pointer will be
    // different to other empty string pointers.  (In general, if you need
    // a pointer to an empty string call CStringStore::getEmpty() instead of
    // doing this.)
    ml::core::CStoredStringPtr UNSET =
        ml::core::CStoredStringPtr::makeStoredString(std::string());
    ml::core::CStoredStringPtr PARTITION_A = ml::model::CStringStore::names().get("pA");
    ml::core::CStoredStringPtr PARTITION_B = ml::model::CStringStore::names().get("pB");
    ml::core::CStoredStringPtr PARTITION_C = ml::model::CStringStore::names().get("pC");

    ml::core::CStoredStringPtr PARTITION_VALUE_1 =
        ml::model::CStringStore::names().get("v1");
    ml::core::CStoredStringPtr PARTITION_VALUE_2 =
        ml::model::CStringStore::names().get("v2");
    ml::core::CStoredStringPtr PARTITION_VALUE_3 =
        ml::model::CStringStore::names().get("v3");

    TestNode root("root");

    ml::model::hierarchical_results_detail::SResultSpec spec;
    spec.s_PartitionFieldName = PARTITION_A;
    spec.s_PartitionFieldValue = PARTITION_VALUE_1;
    ml::model::SAnnotatedProbability emptyAnnotatedProb;

    ml::model::hierarchical_results_detail::SResultSpec unsetSpec;

    CConcreteHierarchicalResultsLevelSet::TNode parent(unsetSpec, emptyAnnotatedProb);
    CConcreteHierarchicalResultsLevelSet::TNode child(spec, emptyAnnotatedProb);
    CConcreteHierarchicalResultsLevelSet::TNode node(spec, emptyAnnotatedProb);
    node.s_Parent = &parent;
    node.s_Children.push_back(&child);

    std::vector<TestNode*> result;

    // without per partition normalization
    {
        CConcreteHierarchicalResultsLevelSet levelSet(root);
        levelSet.elements(node, false, CTestNodeFactory(), result, false);
        std::for_each(result.begin(), result.end(), print);
        CPPUNIT_ASSERT_EQUAL(size_t(1), result.size());
        CPPUNIT_ASSERT_EQUAL(std::string("pA"), result[0]->s_Name);
    }

    // with per partition normalization
    {
        CConcreteHierarchicalResultsLevelSet levelSet(root);
        levelSet.elements(node, false, CTestNodeFactory(), result, true);

        CPPUNIT_ASSERT_EQUAL(size_t(1), result.size());
        CPPUNIT_ASSERT_EQUAL(std::string("pAv1"), result[0]->s_Name);

        ml::model::hierarchical_results_detail::SResultSpec specB;
        specB.s_PartitionFieldName = PARTITION_B;
        specB.s_PartitionFieldValue = PARTITION_VALUE_1;

        CConcreteHierarchicalResultsLevelSet::TNode nodeB(specB, emptyAnnotatedProb);
        nodeB.s_Parent = &parent;
        nodeB.s_Children.push_back(&child);

        levelSet.elements(nodeB, false, CTestNodeFactory(), result, true);

        std::for_each(result.begin(), result.end(), print);
        CPPUNIT_ASSERT_EQUAL(size_t(1), result.size());
        CPPUNIT_ASSERT_EQUAL(std::string("pBv1"), result[0]->s_Name);
    }
}
