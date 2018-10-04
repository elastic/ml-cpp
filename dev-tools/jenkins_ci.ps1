function Exec {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$cmd,
        [string]$errorMessage = ($msgs.error_bad_command -f $cmd)
    )

    try {
        $global:lastexitcode = 0
        & $cmd
        if ($lastexitcode -ne 0) {
            throw $errorMessage
        }
    }
    catch [Exception] {
        throw $_
    }
}

# Generate a Vault token
$env:VAULT_TOKEN = & vault write -field=token auth/approle/login role_id="$env:VAULT_ROLE_ID" secret_id="$env:VAULT_SECRET_ID"

# Load aws-creds from vault
$aws_creds = & vault read -format=json -field=data aws-dev/creds/prelertartifacts
$env:ML_AWS_ACCESS_KEY=(echo $aws_creds | jq -r ".access_key")
$env:ML_AWS_SECRET_KEY=(echo $aws_creds | jq -r ".secret_key")

# Remove VAULT_* values
Remove-Item env:VAULT_TOKEN
Remove-Item env:VAULT_ROLE_ID
Remove-Item env:VAULT_SECRET_ID

# Call dev-tools\ci.bat
exec { cmd /c "dev-tools\ci.bat" } "CI Test FAILURE"
