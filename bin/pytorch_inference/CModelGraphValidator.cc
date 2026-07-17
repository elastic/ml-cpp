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

    if (nodeCount > MAX_NODE_COUNT) {
        LOG_ERROR(<< "Model graph is too large: " << nodeCount
                  << " nodes exceeds limit of " << MAX_NODE_COUNT);
        return {false, {}, {}, nodeCount};
    }

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

    // Two-pass check: forbidden ops first, then unrecognised.  This lets us
    // fail fast when a known-dangerous operation is present and avoids the
    // cost of scanning for unrecognised ops on a model we will reject anyway.
    for (const auto& op : observedOps) {
        if (forbiddenOps.contains(op)) {
            result.s_IsValid = false;
            result.s_ForbiddenOps.push_back(op);
        }
    }

    if (result.s_ForbiddenOps.empty()) {
        for (const auto& op : observedOps) {
            if (allowedOps.contains(op) == false) {
                result.s_IsValid = false;
                result.s_UnrecognisedOps.push_back(op);
            }
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
        if (++nodeCount > MAX_NODE_COUNT) {
            return;
        }
        ops.emplace(node->kind().toQualString());
        for (const auto* subBlock : node->blocks()) {
            collectBlockOps(*subBlock, ops, nodeCount);
            if (nodeCount > MAX_NODE_COUNT) {
                return;
            }
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
        if (nodeCount > MAX_NODE_COUNT) {
            return;
        }
    }
}
}
}
