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

#include <CModelGraphValidator.h>

#include <CSupportedOperations.h>

#include <boost/test/unit_test.hpp>

#include <torch/script.h>

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

BOOST_AUTO_TEST_CASE(testCallMethodForbiddenAfterInlining) {
    // prim::CallMethod must not appear after graph inlining; its presence
    // means a method call could not be resolved and the graph cannot be
    // fully validated.
    TStringSet observed{"aten::linear", "prim::Constant", "prim::CallMethod"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(1, result.s_ForbiddenOps.size());
    BOOST_REQUIRE_EQUAL("prim::CallMethod", result.s_ForbiddenOps[0]);
    BOOST_REQUIRE(result.s_UnrecognisedOps.empty());
}

BOOST_AUTO_TEST_CASE(testCallFunctionForbiddenAfterInlining) {
    TStringSet observed{"aten::linear", "prim::CallFunction"};

    auto result = CModelGraphValidator::validate(
        observed, CSupportedOperations::ALLOWED_OPERATIONS,
        CSupportedOperations::FORBIDDEN_OPERATIONS);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE_EQUAL(1, result.s_ForbiddenOps.size());
    BOOST_REQUIRE_EQUAL("prim::CallFunction", result.s_ForbiddenOps[0]);
}

BOOST_AUTO_TEST_CASE(testMaxNodeCountConstant) {
    BOOST_REQUIRE(CModelGraphValidator::MAX_NODE_COUNT > 0);
    BOOST_REQUIRE_EQUAL(std::size_t{1000000}, CModelGraphValidator::MAX_NODE_COUNT);
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

// --- Integration tests using real TorchScript modules ---

BOOST_AUTO_TEST_CASE(testValidModuleWithAllowedOps) {
    // A simple module using only aten::add and aten::mul, both of which
    // are in the allowed set.
    ::torch::jit::Module m("__torch__.ValidModel");
    m.define(R"(
        def forward(self, x: Tensor) -> Tensor:
            return x + x * x
    )");

    auto result = CModelGraphValidator::validate(m);

    BOOST_REQUIRE(result.s_IsValid);
    BOOST_REQUIRE(result.s_ForbiddenOps.empty());
    BOOST_REQUIRE(result.s_UnrecognisedOps.empty());
    BOOST_REQUIRE(result.s_NodeCount > 0);
}

BOOST_AUTO_TEST_CASE(testModuleWithUnrecognisedOps) {
    // torch.sin is not in the transformer allowlist.
    ::torch::jit::Module m("__torch__.UnknownOps");
    m.define(R"(
        def forward(self, x: Tensor) -> Tensor:
            return torch.sin(x)
    )");

    auto result = CModelGraphValidator::validate(m);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE(result.s_ForbiddenOps.empty());
    BOOST_REQUIRE(result.s_UnrecognisedOps.empty() == false);
    bool foundSin = false;
    for (const auto& op : result.s_UnrecognisedOps) {
        if (op == "aten::sin") {
            foundSin = true;
        }
    }
    BOOST_REQUIRE(foundSin);
}

BOOST_AUTO_TEST_CASE(testModuleNodeCountPopulated) {
    ::torch::jit::Module m("__torch__.NodeCount");
    m.define(R"(
        def forward(self, x: Tensor) -> Tensor:
            a = x + x
            b = a * a
            c = b - a
            return c
    )");

    auto result = CModelGraphValidator::validate(m);

    BOOST_REQUIRE(result.s_NodeCount > 0);
}

BOOST_AUTO_TEST_CASE(testModuleWithSubmoduleInlines) {
    // Create a parent module with a child submodule. After inlining,
    // the child's operations should be visible and validated.
    ::torch::jit::Module child("__torch__.Child");
    child.define(R"(
        def forward(self, x: Tensor) -> Tensor:
            return torch.sin(x)
    )");

    ::torch::jit::Module parent("__torch__.Parent");
    parent.register_module("child", child);
    parent.define(R"(
        def forward(self, x: Tensor) -> Tensor:
            return self.child.forward(x) + x
    )");

    auto result = CModelGraphValidator::validate(parent);

    BOOST_REQUIRE(result.s_IsValid == false);
    bool foundSin = false;
    for (const auto& op : result.s_UnrecognisedOps) {
        if (op == "aten::sin") {
            foundSin = true;
        }
    }
    BOOST_REQUIRE(foundSin);
}

// --- Integration tests with malicious .pt model fixtures ---
//
// These load real TorchScript models that simulate attack vectors.
// The .pt files are generated by testfiles/generate_malicious_models.py.

namespace {
bool hasForbiddenOp(const CModelGraphValidator::SResult& result, const std::string& op) {
    return std::find(result.s_ForbiddenOps.begin(), result.s_ForbiddenOps.end(),
                     op) != result.s_ForbiddenOps.end();
}

bool hasUnrecognisedOp(const CModelGraphValidator::SResult& result, const std::string& op) {
    return std::find(result.s_UnrecognisedOps.begin(), result.s_UnrecognisedOps.end(),
                     op) != result.s_UnrecognisedOps.end();
}
}

BOOST_AUTO_TEST_CASE(testMaliciousFileReader) {
    // A model that uses aten::from_file to read arbitrary files.
    auto module = ::torch::jit::load("testfiles/malicious_models/malicious_file_reader.pt");
    auto result = CModelGraphValidator::validate(module);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE(hasForbiddenOp(result, "aten::from_file"));
}

BOOST_AUTO_TEST_CASE(testMaliciousMixedFileReader) {
    // A model that mixes allowed ops (aten::add) with a forbidden
    // aten::from_file. The entire model must be rejected.
    auto module = ::torch::jit::load("testfiles/malicious_models/malicious_mixed_file_reader.pt");
    auto result = CModelGraphValidator::validate(module);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE(hasForbiddenOp(result, "aten::from_file"));
    BOOST_REQUIRE(result.s_UnrecognisedOps.empty());
}

BOOST_AUTO_TEST_CASE(testMaliciousHiddenInSubmodule) {
    // Unrecognised ops buried three levels deep in nested submodules.
    // The validator must inline through all submodules to find them.
    auto module = ::torch::jit::load("testfiles/malicious_models/malicious_hidden_in_submodule.pt");
    auto result = CModelGraphValidator::validate(module);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE(result.s_ForbiddenOps.empty());
    BOOST_REQUIRE(hasUnrecognisedOp(result, "aten::sin"));
}

BOOST_AUTO_TEST_CASE(testMaliciousConditionalBranch) {
    // An unrecognised op hidden inside a conditional branch. The
    // validator must recurse into prim::If blocks to detect it.
    auto module = ::torch::jit::load("testfiles/malicious_models/malicious_conditional.pt");
    auto result = CModelGraphValidator::validate(module);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE(hasUnrecognisedOp(result, "aten::sin"));
}

BOOST_AUTO_TEST_CASE(testMaliciousManyUnrecognisedOps) {
    // A model using many different unrecognised ops (sin, cos, tan, exp).
    auto module = ::torch::jit::load("testfiles/malicious_models/malicious_many_unrecognised.pt");
    auto result = CModelGraphValidator::validate(module);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE(result.s_ForbiddenOps.empty());
    BOOST_REQUIRE(result.s_UnrecognisedOps.size() >= 4);
    BOOST_REQUIRE(hasUnrecognisedOp(result, "aten::sin"));
    BOOST_REQUIRE(hasUnrecognisedOp(result, "aten::cos"));
    BOOST_REQUIRE(hasUnrecognisedOp(result, "aten::tan"));
    BOOST_REQUIRE(hasUnrecognisedOp(result, "aten::exp"));
}

BOOST_AUTO_TEST_CASE(testMaliciousFileReaderInSubmodule) {
    // The forbidden aten::from_file is hidden inside a submodule.
    // After inlining, the validator must still detect it.
    auto module = ::torch::jit::load("testfiles/malicious_models/malicious_file_reader_in_submodule.pt");
    auto result = CModelGraphValidator::validate(module);

    BOOST_REQUIRE(result.s_IsValid == false);
    BOOST_REQUIRE(hasForbiddenOp(result, "aten::from_file"));
}

BOOST_AUTO_TEST_SUITE_END()
