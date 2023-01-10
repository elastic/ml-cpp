#!/bin/bash

github_org="elastic"
repo_name="ml-cpp"

message=$(vault read -field=message secret/ci/${github_org}-${repo_name}/test)

echo "Reading the Vault secret..."
echo "$message"
echo

echo "Checking token values..."
token_lookup=$(vault token lookup | grep -E 'policies|meta')
echo "$token_lookup"
