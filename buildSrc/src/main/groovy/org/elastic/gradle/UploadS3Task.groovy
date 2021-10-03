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
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.TaskAction
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import javax.inject.Inject

/**
 * A task to upload files to s3, which allows delayed resolution of the s3 path
 */
class UploadS3Task extends DefaultTask {

    private Map<File, Object> toUpload = new LinkedHashMap<>()

    @Input
    String bucket

    UploadS3Task() {
        ext.set('needs.aws', true)
    }

    /**
     * Add a file to be uploaded to s3. The key object will be evaluated at runtime.
     *
     * If file is a directory, all files in the directory will be uploaded to the key as a prefix.
     */
    public void upload(File file, Object key) {
        toUpload.put(file, key)
    }

    @TaskAction
    public void uploadToS3() {

        Region region = Region.EU_WEST_1
        AwsCredentials creds = AwsBasicCredentials.create(project.mlAwsAccessKey, project.mlAwsSecretKey)
        AwsCredentialsProvider credsProvider = StaticCredentialsProvider.create(creds)
        S3Client client = S3Client.builder().region(region).credentialsProvider(credsProvider).build()

        for (Map.Entry<File, Object> entry : toUpload) {
            File file = entry.getKey()
            String key = entry.getValue().toString()
            if (file.isDirectory()) {
                uploadDir(client, file, key)
            } else {
                uploadFile(client, file, key)
            }
        }
    }

    /** Recursively upload all files in a directory. */
    private void uploadDir(S3Client client, File dir, String prefix) {
        for (File subfile : dir.listFiles()) {
            if (subfile.isDirectory()) {
                uploadDir(client, subfile, "${prefix}/${subfile.name}")
            } else {
                String subkey = "${prefix}/${subfile.name}"
                uploadFile(client, subfile, subkey)
            }
        }
    }

    /** Upload a single file */
    private void uploadFile(S3Client client, File file, String key) {
        logger.info("Uploading ${file.name} to ${key}")
        PutObjectRequest objectRequest = PutObjectRequest.builder().bucket(bucket).key(key).build()
        client.putObject(objectRequest, file.toPath())
    }
}
