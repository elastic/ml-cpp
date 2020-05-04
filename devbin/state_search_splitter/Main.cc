/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
//! \brief
//! Reformat searches of model state to assist with problem reproduction.
//!
//! DESCRIPTION:\n
//! Utility to take the output of a search of a .ml-state index that
//! returns all state documents for a given model snapshot and reformat
//! it as the multiple chunks that are streamed to the autodetect
//! process (separated by \0 characters).
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone dev program, not shipped with the product.
//!
#include <core/CLogger.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {
const std::string ID_PREFIX{"\"_id\":\""};
const std::string SOURCE_PREFIX{"\"_source\":"};
}

void skipPreamble(std::istream& input) {
    std::string preamble;
    char c;
    while (input.get(c)) {
        preamble += c;
        if (c == '[') {
            break;
        }
    }
    LOG_DEBUG(<< "Preamble is: " << preamble);
}

void saveSource(const std::string& doc) {

    std::size_t idPos{doc.find(ID_PREFIX)};
    if (idPos == std::string::npos) {
        LOG_ERROR(<< "_id start not found");
        return;
    }
    std::size_t idEnd{doc.find('"', idPos + ID_PREFIX.length())};
    if (idEnd == std::string::npos) {
        LOG_ERROR(<< "_id end not found");
        return;
    }
    std::size_t sourcePos{doc.find(SOURCE_PREFIX)};
    if (sourcePos == std::string::npos) {
        LOG_ERROR(<< "_source start not found");
        return;
    }
    std::size_t sourceEnd{doc.length() - 1};
    std::string docId{doc.substr(idPos + ID_PREFIX.length(),
                                 idEnd - idPos - ID_PREFIX.length())};
    std::ofstream singleSourceFile{docId + ".json"};
    if (singleSourceFile.is_open() == false) {
        LOG_ERROR(<< "Could not open output file " << docId << ".json");
        return;
    }
    LOG_INFO(<< "Saving _source of document with _id " << docId);
    singleSourceFile << doc.substr(sourcePos + SOURCE_PREFIX.length(),
                                   sourceEnd - sourcePos - SOURCE_PREFIX.length())
                     << std::endl;
}

void extractDocs(std::istream& input) {
    int level{0};
    char c;
    std::string currentDoc;
    while (input.get(c)) {
        if (c == '{') {
            ++level;
            if (level == 1) {
                currentDoc.clear();
            }
            currentDoc += c;
        } else if (c == '}') {
            currentDoc += c;
            --level;
            if (level == 0) {
                saveSource(currentDoc);
            }
        } else {
            if (level > 0) {
                currentDoc += c;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Utility to take the output of a search of a .ml-state index\n"
                     "that returns all state documents for a given model snapshot\n"
                     "and reformat it as the multiple chunks that are streamed\n"
                     "to the autodetect process (separated by \\0 characters)\n";
        std::cerr << "Usage: " << argv[0] << " <input file>" << std::endl;
        return EXIT_FAILURE;
    }
    LOG_INFO(<< "Opening input file: " << argv[1]);
    std::ifstream inputFile(argv[1]);
    if (inputFile.is_open() == false) {
        LOG_ERROR(<< "Could not open input file " << argv[1]);
        return EXIT_FAILURE;
    }
    skipPreamble(inputFile);
    extractDocs(inputFile);
    return inputFile.bad() ? EXIT_FAILURE : EXIT_SUCCESS;
}
