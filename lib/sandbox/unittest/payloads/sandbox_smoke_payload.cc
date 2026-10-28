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

// Deliberately dependency-free sandboxee for CSandboxForkserverSmokeTest_Linux.
// It exists only to prove the vendored Sandbox2 forkserver can fork, exec,
// and reap a child through the full pipeline patched in
// 3rd_party/patches/sandboxed-api/0004-forkserver-zlib-static-libstdcxx.patch.
// It carries no ml-cpp library dependencies and no sandbox policy of its
// own - policy design is out of scope for this dormant dependency
// foundation and lands in a follow-up PR.

#include <cstdio>
#include <cstdlib>

int main() {
    std::printf("sandbox2-smoke-ok\n");
    return EXIT_SUCCESS;
}
