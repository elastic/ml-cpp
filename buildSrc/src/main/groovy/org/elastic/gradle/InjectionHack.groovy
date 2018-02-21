/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.elastic.gradle

import org.gradle.api.DefaultTask
import org.gradle.internal.logging.progress.ProgressLoggerFactory

import javax.inject.Inject

/**
 * A dummy task that exists only so we can grab a progress logger factory, because injection is broken with extensions.
 */
public class InjectionHack extends DefaultTask {
    public InjectionHack() {
        // add as a dependency so this doesn't show up in task list
        project.clean.dependsOn(this)
    }

    @Inject
    ProgressLoggerFactory getProgressLoggerFactory() {
        throw new UnsupportedOperationException()
    }
}
