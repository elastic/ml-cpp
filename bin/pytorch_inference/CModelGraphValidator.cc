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

#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/read_adapter_interface.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ml {
namespace torch {
namespace {

//! Minimal in-memory ReadAdapterInterface so a PyTorchStreamReader can parse an
//! archive that is already fully buffered in memory, without taking ownership.
class CMemoryReadAdapter final : public caffe2::serialize::ReadAdapterInterface {
public:
    CMemoryReadAdapter(const char* data, std::size_t size)
        : m_Data{data}, m_Size{size} {}

    std::size_t size() const override { return m_Size; }

    std::size_t
    read(std::uint64_t pos, void* buf, std::size_t n, const char* /*what*/ = "") const override {
        if (pos >= m_Size) {
            return 0;
        }
        n = std::min(n, m_Size - static_cast<std::size_t>(pos));
        std::memcpy(buf, m_Data + pos, n);
        return n;
    }

private:
    const char* m_Data;
    std::size_t m_Size;
};

//! True for the serialised TorchScript source records (code/**/*.py) that hold
//! the executable code of the module and its submodules — including
//! __setstate__.  Other records (data.pkl, constants.pkl, version, tensors) are
//! not executable source and are skipped.
bool isSerialisedCodeRecord(const std::string& name) {
    constexpr std::string_view PY_EXT{".py"};
    if (name.size() < PY_EXT.size() ||
        name.compare(name.size() - PY_EXT.size(), PY_EXT.size(), PY_EXT) != 0) {
        return false;
    }
    // Archive entries are typically "<root>/code/....py"; accept a leading
    // "code/" too in case the root prefix is absent.
    return name.find("/code/") != std::string::npos || name.rfind("code/", 0) == 0;
}
}

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

CModelGraphValidator::TStringVec
CModelGraphValidator::scanSerialisedCodeForForbiddenOps(const char* data, std::size_t size) {
    TStringSet found;

    // Build the textual signatures we are looking for in serialised TorchScript
    // source.  Known stable forms:
    //   aten::foo            -> torch.foo(
    //   inductor::_reinterpret_tensor -> ops.inductor._reinterpret_tensor(
    // prim::CallFunction / prim::CallMethod describe unresolved calls in an
    // inlined graph and have no stable textual form in the serialised source.
    std::vector<std::pair<std::string, std::string>> signatures; // (needle, qualified op)
    constexpr std::string_view ATEN{"aten::"};
    constexpr std::string_view INDUCTOR{"inductor::"};
    for (const auto& op : CSupportedOperations::FORBIDDEN_OPERATIONS) {
        if (op.size() > ATEN.size() && op.substr(0, ATEN.size()) == ATEN) {
            std::string shortName{op.substr(ATEN.size())};
            signatures.emplace_back("torch." + shortName + "(", std::string{op});
        } else if (op.size() > INDUCTOR.size() && op.substr(0, INDUCTOR.size()) == INDUCTOR) {
            std::string shortName{op.substr(INDUCTOR.size())};
            signatures.emplace_back("ops.inductor." + shortName + "(", std::string{op});
        }
    }

    try {
        auto adapter = std::make_shared<CMemoryReadAdapter>(data, size);
        caffe2::serialize::PyTorchStreamReader reader{adapter};

        for (const auto& name : reader.getAllRecords()) {
            if (isSerialisedCodeRecord(name) == false) {
                continue;
            }
            auto[recordData, recordSize] = reader.getRecord(name);
            std::string_view code{static_cast<const char*>(recordData.get()), recordSize};
            for (const auto & [ needle, qualifiedOp ] : signatures) {
                if (code.find(needle) != std::string_view::npos) {
                    found.emplace(qualifiedOp);
                }
            }
        }
    } catch (const std::exception& e) {
        // If the archive cannot be parsed as a stream we do not treat this as a
        // rejection here: torch::jit::load will attempt the same parse and
        // surface a clear error.  The post-load graph validator still applies.
        LOG_WARN(<< "Pre-load model code scan skipped (could not parse archive): "
                 << e.what());
        return {};
    }

    TStringVec result{found.begin(), found.end()};
    std::sort(result.begin(), result.end());
    return result;
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
