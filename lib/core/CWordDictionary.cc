/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CWordDictionary.h>

#include <core/CLogger.h>
#include <core/CResourceLocator.h>
#include <core/CScopedFastLock.h>
#include <core/CStrCaseCmp.h>
#include <core/CStringUtils.h>

#include <fstream>

#include <ctype.h>

namespace ml {
namespace core {

namespace {

const char PART_OF_SPEECH_SEPARATOR('@');

CWordDictionary::EPartOfSpeech partOfSpeechFromCode(char partOfSpeechCode) {
    // These codes are taken from the readme file that comes with Moby
    // Part-of-Speech - see http://icon.shef.ac.uk/Moby/mpos.html
    switch (partOfSpeechCode) {
    case '?':
        // This means the word existed in SCOWL but not Moby and none of the
        // heuristics in crossref.py worked
        return CWordDictionary::E_UnknownPart;
    case 'N':
    case 'h': // Currently don't distinguish noun phrases
    case 'o': // Currently don't distinguish nominative nouns
        return CWordDictionary::E_Noun;
    case 'p':
        return CWordDictionary::E_Plural;
    case 'V':
    case 't': // Currently don't distinguish transitive verbs
    case 'i': // Currently don't distinguish intransitive verbs
        return CWordDictionary::E_Verb;
    case 'A':
        return CWordDictionary::E_Adjective;
    case 'v':
        return CWordDictionary::E_Adverb;
    case 'C':
        return CWordDictionary::E_Conjunction;
    case 'P':
        return CWordDictionary::E_Preposition;
    case '!':
        return CWordDictionary::E_Interjection;
    case 'r':
        return CWordDictionary::E_Pronoun;
    case 'D':
        return CWordDictionary::E_DefiniteArticle;
    case 'I':
        return CWordDictionary::E_IndefiniteArticle;
    default:
        break;
    }

    // This should be treated as an error when returned by this function
    return CWordDictionary::E_NotInDictionary;
}
}

const char* CWordDictionary::DICTIONARY_FILE("ml-en.dict");

CFastMutex CWordDictionary::ms_LoadMutex;
volatile CWordDictionary* CWordDictionary::ms_Instance(0);

const CWordDictionary& CWordDictionary::instance() {
    if (ms_Instance == 0) {
        CScopedFastLock lock(ms_LoadMutex);

        // Even if we get into this code block in more than one thread, whatever
        // measures the compiler is taking to ensure this variable is only
        // constructed once should be fine given that the block is protected by
        // a mutex.
        static volatile CWordDictionary instance;

        ms_Instance = &instance;
    }

    // Need to explicitly cast away volatility
    return *const_cast<const CWordDictionary*>(ms_Instance);
}

bool CWordDictionary::isInDictionary(const std::string& str) const {
    return m_DictionaryWords.find(str) != m_DictionaryWords.end();
}

CWordDictionary::EPartOfSpeech CWordDictionary::partOfSpeech(const std::string& str) const {
    TStrUMapCItr iter = m_DictionaryWords.find(str);
    if (iter == m_DictionaryWords.end()) {
        return E_NotInDictionary;
    }
    return iter->second;
}

CWordDictionary::CWordDictionary() {
    std::string fileToLoad(CResourceLocator::resourceDir() + '/' + DICTIONARY_FILE);

    // If the file can't be read for some reason, we just end up with an empty
    // dictionary
    std::ifstream ifs(fileToLoad.c_str());
    if (ifs.is_open()) {
        LOG_DEBUG("Populating word dictionary from file " << fileToLoad);

        std::string word;
        while (std::getline(ifs, word)) {
            CStringUtils::trimWhitespace(word);
            if (word.empty()) {
                continue;
            }
            size_t sepPos(word.find(PART_OF_SPEECH_SEPARATOR));
            if (sepPos == std::string::npos) {
                LOG_ERROR("Found word with no part-of-speech separator: " << word);
                continue;
            }
            if (sepPos == 0) {
                LOG_ERROR("Found part-of-speech separator with no preceding word: " << word);
                continue;
            }
            if (sepPos + 1 >= word.length()) {
                LOG_ERROR("Found word with no part-of-speech code: " << word);
                continue;
            }
            char partOfSpeechCode(word[sepPos + 1]);
            EPartOfSpeech partOfSpeech(partOfSpeechFromCode(partOfSpeechCode));
            if (partOfSpeech == E_NotInDictionary) {
                LOG_ERROR("Unknown part-of-speech code (" << partOfSpeechCode << ") for word: " << word);
                continue;
            }
            word.erase(sepPos);
            m_DictionaryWords[word] = partOfSpeech;
        }

        LOG_DEBUG("Populated word dictionary with " << m_DictionaryWords.size() << " words");
    } else {
        LOG_ERROR("Failed to open dictionary file " << fileToLoad);
    }
}

CWordDictionary::~CWordDictionary() {
    ms_Instance = 0;
}

size_t CWordDictionary::CStrHashIgnoreCase::operator()(const std::string& str) const {
    size_t hash(0);

    for (std::string::const_iterator iter = str.begin(); iter != str.end(); ++iter) {
        hash *= 17;
        hash += ::tolower(*iter);
    }

    return hash;
}

bool CWordDictionary::CStrEqualIgnoreCase::operator()(const std::string& lhs, const std::string& rhs) const {
    return lhs.length() == rhs.length() && CStrCaseCmp::strCaseCmp(lhs.c_str(), rhs.c_str()) == 0;
}
}
}
