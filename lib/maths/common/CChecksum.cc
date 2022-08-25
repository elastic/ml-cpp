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

#include <maths/common/CChecksum.h>

#include <core/CIEEE754.h>
#include <core/CStoredStringPtr.h>

#include <cstdio>
#include <cstring>
#include <functional>

namespace ml {
namespace maths {
namespace common {
namespace checksum_detail {
namespace {
const std::hash<std::vector<bool>> vectorBoolHasher;
}

std::uint64_t CChecksumImpl<BasicChecksum>::dispatch(std::uint64_t seed, double target) {
    // A fuzzy checksum implementation is useful for floating point values
    // so we know we're close to a reasonable precision. This checksums the
    // printed value so that it's stable over persist and restore.
    target = core::CIEEE754::round(target, core::CIEEE754::E_SinglePrecision);
    char buf[4 * sizeof(double)];
    std::memset(buf, 0, sizeof(buf));
    std::sprintf(buf, "%.7g", target);
    return core::CHashing::safeMurmurHash64(&buf[0], 4 * sizeof(double), seed);
}

std::uint64_t CChecksumImpl<BasicChecksum>::dispatch(
    std::uint64_t seed,
    const core::CHashing::CUniversalHash::CUInt32UnrestrictedHash& target) {
    seed = core::CHashing::hashCombine(seed, static_cast<std::uint64_t>(target.a()));
    return core::CHashing::hashCombine(seed, static_cast<std::uint64_t>(target.b()));
}

std::uint64_t CChecksumImpl<BasicChecksum>::dispatch(std::uint64_t seed,
                                                     const std::string& target) {
    return core::CHashing::safeMurmurHash64(target.data(),
                                            static_cast<int>(target.size()), seed);
}

std::uint64_t CChecksumImpl<BasicChecksum>::dispatch(std::uint64_t seed,
                                                     const core::CStoredStringPtr& target) {
    return target == nullptr ? seed : dispatch(seed, *target);
}

std::uint64_t CChecksumImpl<ContainerChecksum>::dispatch(std::uint64_t seed,
                                                         const std::vector<bool>& target) {
    return core::CHashing::hashCombine(seed, vectorBoolHasher(target));
}

std::uint64_t CChecksumImpl<ContainerChecksum>::dispatch(std::uint64_t seed,
                                                         const std::string& target) {
    return CChecksumImpl<BasicChecksum>::dispatch(seed, target);
}
}
}
}
}
