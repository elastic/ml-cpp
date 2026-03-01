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

#include <algorithm>

namespace ml {
namespace torch {

CModelGraphValidator::SResult CModelGraphValidator::validate(const ::torch::jit::Module& module) {

    TStringSet observedOps;
    collectModuleOps(module, observedOps);

    return validate(observedOps, CSupportedOperations::ALLOWED_OPERATIONS,
                    CSupportedOperations::FORBIDDEN_OPERATIONS);
}

CModelGraphValidator::SResult
CModelGraphValidator::validate(const TStringSet& observedOps,
                               const std::unordered_set<std::string_view>& allowedOps,
                               const std::unordered_set<std::string_view>& forbiddenOps) {

    SResult result;

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

void CModelGraphValidator::collectBlockOps(const ::torch::jit::Block& block, TStringSet& ops) {
    for (const auto* node : block.nodes()) {
        ops.emplace(node->kind().toQualString());
        for (const auto* subBlock : node->blocks()) {
            collectBlockOps(*subBlock, ops);
        }
    }
}

void CModelGraphValidator::collectModuleOps(const ::torch::jit::Module& module,
                                            TStringSet& ops) {
    for (const auto& method : module.get_methods()) {
        auto graph = method.graph();
        collectBlockOps(*graph->block(), ops);
    }

    for (const auto& child : module.children()) {
        collectModuleOps(child, ops);
    }
}
}
}
