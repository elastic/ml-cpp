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

#include "../CModelGraphValidator.h"

#include "../CSupportedOperations.h"

#include <boost/test/unit_test.hpp>

#include <string>
#include <string_view>
#include <unordered_set>

using namespace ml::torch;
using TStringSet = CModelGraphValidator::TStringSet;
using TStringViewSet = std::unordered_set<std::string_view>;

BOOST_AUTO_TEST_SUITE(CModelGraphValidatorTest)

BOOST_AUTO_TEST_CASE(testAllAllowedOpsPass) {
    // A model using only allowed ops should pass validation.
    TStringSet observed{"aten::linear",    "aten::layer_norm", "aten::gelu",
                        "aten::embedding", "prim::Constant",   "prim::GetAttr"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid);
    BOOST_REQUIRE(result.s_ForbiddenOps.empty());
    BOOST_REQUIRE(result.s_UnrecognisedOps.empty());
}

BOOST_AUTO_TEST_CASE(testEmptyGraphPasses) {
    TStringSet observed;

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid);
    BOOST_REQUIRE(result.s_ForbiddenOps.empty());
    BOOST_REQUIRE(result.s_UnrecognisedOps.empty());
}

BOOST_AUTO_TEST_CASE(testForbiddenOpsRejected) {
    TStringSet observed{"aten::linear", "aten::from_file", "prim::Constant"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(1, result.s_ForbiddenOps.size());
    BOOST_REQUIRE_EQUAL("aten::from_file", result.s_ForbiddenOps[0]);
    BOOST_REQUIRE(result.s_UnrecognisedOps.empty());
}

BOOST_AUTO_TEST_CASE(testMultipleForbiddenOps) {
    TStringSet observed{"aten::from_file", "aten::save"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(2, result.s_ForbiddenOps.size());
    BOOST_REQUIRE_EQUAL("aten::from_file", result.s_ForbiddenOps[0]);
    BOOST_REQUIRE_EQUAL("aten::save", result.s_ForbiddenOps[1]);
}

BOOST_AUTO_TEST_CASE(testUnrecognisedOpsRejected) {
    TStringSet observed{"aten::linear", "custom::evil_op", "prim::Constant"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE(result.s_ForbiddenOps.empty());
    BOOST_REQUIRE_EQUAL(1, result.s_UnrecognisedOps.size());
    BOOST_REQUIRE_EQUAL("custom::evil_op", result.s_UnrecognisedOps[0]);
}

BOOST_AUTO_TEST_CASE(testMixedForbiddenAndUnrecognised) {
    TStringSet observed{"aten::save", "custom::backdoor", "aten::linear"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(1, result.s_ForbiddenOps.size());
    BOOST_REQUIRE_EQUAL("aten::save", result.s_ForbiddenOps[0]);
    BOOST_REQUIRE_EQUAL(1, result.s_UnrecognisedOps.size());
    BOOST_REQUIRE_EQUAL("custom::backdoor", result.s_UnrecognisedOps[0]);
}

BOOST_AUTO_TEST_CASE(testResultsSorted) {
    TStringSet observed{"zzz::unknown", "aaa::unknown", "mmm::unknown"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(3, result.s_UnrecognisedOps.size());
    BOOST_REQUIRE_EQUAL("aaa::unknown", result.s_UnrecognisedOps[0]);
    BOOST_REQUIRE_EQUAL("mmm::unknown", result.s_UnrecognisedOps[1]);
    BOOST_REQUIRE_EQUAL("zzz::unknown", result.s_UnrecognisedOps[2]);
}

BOOST_AUTO_TEST_CASE(testTypicalBertOps) {
    // Simulate a realistic BERT-like op set.
    TStringSet observed{"aten::Int",
                        "aten::ScalarImplicit",
                        "aten::__and__",
                        "aten::add",
                        "aten::arange",
                        "aten::contiguous",
                        "aten::div",
                        "aten::dropout",
                        "aten::embedding",
                        "aten::expand",
                        "aten::gelu",
                        "aten::ge",
                        "aten::index",
                        "aten::layer_norm",
                        "aten::linear",
                        "aten::masked_fill",
                        "aten::matmul",
                        "aten::mul",
                        "aten::new_ones",
                        "aten::permute",
                        "aten::reshape",
                        "aten::scaled_dot_product_attention",
                        "aten::size",
                        "aten::slice",
                        "aten::softmax",
                        "aten::tanh",
                        "aten::to",
                        "aten::transpose",
                        "aten::unsqueeze",
                        "aten::view",
                        "prim::Constant",
                        "prim::DictConstruct",
                        "prim::GetAttr",
                        "prim::If",
                        "prim::ListConstruct",
                        "prim::NumToTensor",
                        "prim::TupleConstruct"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid);
    BOOST_REQUIRE(result.s_ForbiddenOps.empty());
    BOOST_REQUIRE(result.s_UnrecognisedOps.empty());
}

BOOST_AUTO_TEST_CASE(testCustomAllowlistAndForbiddenList) {
    // Verify the three-argument overload works with arbitrary lists.
    TStringViewSet allowed{"op::a", "op::b", "op::c"};
    TStringViewSet forbidden{"op::bad"};
    TStringSet observed{"op::a", "op::b"};

    auto result = CModelGraphValidator::validate(observed, allowed, forbidden);
    BOOST_REQUIRE(result.s_IsValid);

    observed.emplace("op::bad");
    result = CModelGraphValidator::validate(observed, allowed, forbidden);
    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(1, result.s_ForbiddenOps.size());

    observed.erase("op::bad");
    observed.emplace("op::unknown");
    result = CModelGraphValidator::validate(observed, allowed, forbidden);
    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(1, result.s_UnrecognisedOps.size());
}

BOOST_AUTO_TEST_CASE(testForbiddenOpAlsoInAllowlist) {
    // If an op appears in both forbidden and allowed, forbidden takes precedence.
    TStringViewSet allowed{"aten::from_file", "aten::linear"};
    TStringViewSet forbidden{"aten::from_file"};
    TStringSet observed{"aten::from_file", "aten::linear"};

    auto result = CModelGraphValidator::validate(observed, allowed, forbidden);
    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(1, result.s_ForbiddenOps.size());
    BOOST_REQUIRE_EQUAL("aten::from_file", result.s_ForbiddenOps[0]);
}

BOOST_AUTO_TEST_SUITE_END()
