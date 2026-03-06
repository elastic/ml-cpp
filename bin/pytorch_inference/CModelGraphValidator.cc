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

#include "CModelGraphValidator.h"

#include "CSupportedOperations.h"

#include <core/CLogger.h>

#include <torch/csrc/jit/passes/inliner.h>

#include <algorithm>

namespace ml {
namespace torch {

CModelGraphValidator::SResult CModelGraphValidator::validate(const ::torch::jit::Module& module) {

    TStringSet observedOps;
    std::size_t nodeCount{0};
    collectModuleOps(module, observedOps, nodeCount);

    LOG_DEBUG(<< "Model graph contains " << observedOps.size()
              << " distinct operations across " << nodeCount << " nodes");
    for (const auto& op : observedOps) {
        LOG_DEBUG(<< "  observed op: " << op);
    }

    auto result = validate(observedOps, CSupportedOperations::ALLOWED_OPERATIONS,
                           CSupportedOperations::FORBIDDEN_OPERATIONS);
    result.s_NodeCount = nodeCount;
    return result;
}

CModelGraphValidator::SResult
CModelGraphValidator::validate(const TStringSet& observedOps,
                               const std::unordered_set<std::string_view>& allowedOps,
                               const std::unordered_set<std::string_view>& forbiddenOps) {

    SResult result;

    // Check forbidden ops first so they are always reported with a specific
    // error even if they also appear in the allowed set.  See the comment on
    // CSupportedOperations::FORBIDDEN_OPERATIONS for the rationale behind
    // maintaining both a forbidden list and an allowed list.
    for (const auto& op : observedOps) {
        if (forbiddenOps.contains(op)) {
            result.s_IsValid = false;
            result.s_ForbiddenOps.push_back(op);
        } else if (allowedOps.contains(op) == false) {
            result.s_IsValid = false;
            result.s_UnrecognisedOps.push_back(op);
        }
    }

    std::sort(result.s_ForbiddenOps.begin(), result.s_ForbiddenOps.end());
    std::sort(result.s_UnrecognisedOps.begin(), result.s_UnrecognisedOps.end());

    return result;
}

void CModelGraphValidator::collectBlockOps(const ::torch::jit::Block& block,
                                           TStringSet& ops,
                                           std::size_t& nodeCount) {
    for (const auto* node : block.nodes()) {
        ++nodeCount;
        ops.emplace(node->kind().toQualString());
        for (const auto* subBlock : node->blocks()) {
            collectBlockOps(*subBlock, ops, nodeCount);
        }
    }
}

void CModelGraphValidator::collectModuleOps(const ::torch::jit::Module& module,
                                            TStringSet& ops,
                                            std::size_t& nodeCount) {
    for (const auto& method : module.get_methods()) {
        // Inline all method calls so that operations hidden behind
        // prim::CallMethod are surfaced.  After inlining, any remaining
        // prim::CallMethod indicates a call that could not be resolved
        // statically and will be flagged as unrecognised.
        auto graph = method.graph()->copy();
        ::torch::jit::Inline(*graph);
        collectBlockOps(*graph->block(), ops, nodeCount);
    }
}
}
}
