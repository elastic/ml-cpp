#!/usr/bin/env python
#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
#
# Notice: this software, and all information contained
# therein, is the exclusive property of Elasticsearch BV
# and its licensors, if any, and is protected under applicable
# domestic and foreign law, and international treaties.
#
# Reproduction, republication or distribution without the
# express written consent of Elasticsearch BV is
# strictly prohibited.
#

#
# Script for cross-referencing words in scowl.dict with those in mobyposi.txt to
# create en.dict
#
# scowl.dict contains a more reasonably sized word list, but only mobyposi.txt
# has the part-of-speech codes
#
# There are some heuristics to cope with the lack of derived words in
# mobyposi.txt; if none of these work then the type is set to '?'
#

SEPARATOR = '@'

lookupTable = {}

with open('mobyposi.txt', 'r') as mobyFile:
    for line in mobyFile:
        parts = line.strip().split(SEPARATOR)
        if len(parts) == 2:
            word = parts[0].lower()
            partOfSpeechCode = parts[1]
            if not word in lookupTable:
                lookupTable[word] = partOfSpeechCode

with open('scowl.dict', 'r') as scowlFile, open('en.dict', 'w') as mappedFile:
    for line in scowlFile:
        word = line.strip()
        if word in lookupTable:
            mappedFile.write(word)
            mappedFile.write(SEPARATOR)
            origPartOfSpeechCode = lookupTable[word]
            if word[:1] == 'a' and \
               origPartOfSpeechCode[:1] == 'D':
                mappedFile.write(origPartOfSpeechCode.replace('D', 'I', 1))
            else:
                mappedFile.write(origPartOfSpeechCode)
            mappedFile.write('\n')
        elif word[len(word) - 1:] == 's' and \
             word[:len(word) - 1] in lookupTable:
            mappedFile.write(word)
            mappedFile.write(SEPARATOR)
            origPartOfSpeechCode = lookupTable[word[:len(word) - 1]]
            if origPartOfSpeechCode.find('N') == -1:
                mappedFile.write(origPartOfSpeechCode.replace('A', 'p', 1))
            else:
                mappedFile.write(origPartOfSpeechCode.replace('N', '&', 1).replace('A', '&', 1).replace('&', 'p', 1).replace('&', ''))
            mappedFile.write('\n')
        elif word[len(word) - 2:] == 'ed' and \
             (word[:len(word) - 2] in lookupTable or word[:len(word) - 1] in lookupTable):
            mappedFile.write(word)
            mappedFile.write(SEPARATOR)
            if word[:len(word) - 2] in lookupTable:
                origPartOfSpeechCode = lookupTable[word[:len(word) - 2]]
            else:
                origPartOfSpeechCode = lookupTable[word[:len(word) - 1]]
            mappedFile.write(origPartOfSpeechCode.replace('N', '&', 1).replace('p', '&', 1).replace('A', '&', 1).replace('&', 'A', 1).replace('&', ''))
            mappedFile.write('\n')
        elif word[len(word) - 3:] == 'ing' and \
             (word[:len(word) - 3] in lookupTable or word[:len(word) - 3] + 'e' in lookupTable):
            if word[:len(word) - 3] in lookupTable:
                origPartOfSpeechCode = lookupTable[word[:len(word) - 3]]
            else:
                origPartOfSpeechCode = lookupTable[word[:len(word) - 3] + 'e']
            mappedFile.write(word)
            mappedFile.write(SEPARATOR)
            mappedFile.write(origPartOfSpeechCode.replace('N', '&', 1).replace('p', '&', 1).replace('A', '&', 1).replace('&', 'A', 1).replace('&', ''))
            mappedFile.write('\n')
        else:
            mappedFile.write(word)
            mappedFile.write(SEPARATOR)
            if word[len(word) - 1:] == 's' and \
               word[len(word) - 2:] != 'es' and \
               word[len(word) - 2:] != 'ss':
                mappedFile.write('p')
            else:
                mappedFile.write('?')
            mappedFile.write('\n')

