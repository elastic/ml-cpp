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

#ifndef INCLUDED_ml_torch_CModelGraphValidator_h
#define INCLUDED_ml_torch_CModelGraphValidator_h

#include <torch/script.h>

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace ml {
namespace torch {

//! \brief
//! Validates TorchScript model computation graphs against a set of
//! allowed operations.
//!
//! DESCRIPTION:\n
//! Provides defense-in-depth by statically inspecting the TorchScript
//! graph of a loaded model and rejecting any model that contains
//! operations not present in the allowlist derived from supported
//! transformer architectures.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The validation walks all methods of the module and its submodules
//! recursively, collecting every distinct operation.  Any operation
//! that appears in the forbidden set causes immediate rejection.
//! Any operation not in the allowed set is collected and reported.
//! This ensures that even operations buried in helper methods or
//! nested submodules are inspected.
//!
class CModelGraphValidator {
public:
    using TStringSet = std::unordered_set<std::string>;
    using TStringVec = std::vector<std::string>;

    //! Result of validating a model graph.
    struct SResult {
        bool s_IsValid{true};
        TStringVec s_ForbiddenOps;
        TStringVec s_UnrecognisedOps;
        std::size_t s_NodeCount{0};
    };

public:
    //! Validate the computation graph of the given module against the
    //! supported operation allowlist.  Recursively inspects all methods
    //! across all submodules.
    static SResult validate(const ::torch::jit::Module& module);

    //! Validate a pre-collected set of operation names.  Useful for
    //! unit testing the matching logic without requiring a real model.
    static SResult validate(const TStringSet& observedOps,
                            const std::unordered_set<std::string_view>& allowedOps,
                            const std::unordered_set<std::string_view>& forbiddenOps);

private:
    //! Collect all operation names from a block, recursing into sub-blocks.
    static void collectBlockOps(const ::torch::jit::Block& block,
                                TStringSet& ops,
                                std::size_t& nodeCount);

    //! Inline all method calls and collect ops from the flattened graph.
    //! After inlining, prim::CallMethod should not appear; if it does,
    //! the call could not be resolved statically and is treated as
    //! unrecognised.
    static void collectModuleOps(const ::torch::jit::Module& module,
                                 TStringSet& ops,
                                 std::size_t& nodeCount);
};
}
}

#endif // INCLUDED_ml_torch_CModelGraphValidator_h
