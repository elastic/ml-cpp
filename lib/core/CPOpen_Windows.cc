/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CPOpen.h>

#include <string>

namespace ml {
namespace core {

FILE* CPOpen::pOpen(const char* command, const char* mode) {
    if (command == 0) {
        return 0;
    }

    // On Windows, the entire command needs to be quoted, as it's passed as a
    // single argument to the command interpreter, i.e. the process that gets
    // created by _popen() is:
    //
    // cmd.exe /c <ARG>
    //
    // The quoting rules for Microsoft programs are pretty messy compared to
    // those used on Unix - see here for details:
    // http://msdn.microsoft.com/en-us/library/17w5ykft(v=vs.100).aspx
    //
    // However, if we make the assumption that the command we've been passed has
    // already been quoted for cmd.exe by the CShellArgQuoter class, things get
    // a lot simpler, because every double quote will already be preceded by
    // a caret, so there's no danger of a double quote being immediately
    // preceded by a backslash in the underlying string.  Then, we can take
    // advantage of the fact that (from the above link), "A quoted string can
    // be embedded in an argument".
    std::string quoted(1, '"');
    quoted += command;
    quoted += '"';

    return ::_popen(quoted.c_str(), mode);
}

int CPOpen::pClose(FILE* stream) {
    return ::_pclose(stream);
}
}
}
