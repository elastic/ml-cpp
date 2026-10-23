# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Windows counterpart of check_ca_trust.sh (see there): validation errors =
# PASS, validation succeeds = FAIL, connection error = INCONCLUSIVE.  Uses .NET
# SslStream (Schannel on Windows) so the result reflects the OS trust store.
# Exit status: 0 = all pass, 1 = any FAIL, 2 = inconclusive only.

param(
    [string[]]$Hosts = @("dv-g3b-emudhra.we-have-a-problem.com", "dv-g3b-aaa-emudhra.we-have-a-problem.com")
)

$ErrorActionPreference = "Stop"

Write-Output "--- Environment"
Write-Output ([System.Environment]::OSVersion.VersionString)
Write-Output "PowerShell $($PSVersionTable.PSVersion)"

# The validation callback must be compiled: a scriptblock delegate hangs.
Add-Type -TypeDefinition @"
using System;
using System.Net.Security;
using System.Net.Sockets;
using System.Security.Cryptography.X509Certificates;

public static class EmudhraTrustProbe {
    // Returns "PASS|<detail>", "FAIL|" or "INCONCLUSIVE|<detail>".
    public static string Probe(string host) {
        SslPolicyErrors errors = SslPolicyErrors.None;
        string chainStatus = "";
        RemoteCertificateValidationCallback cb = delegate(object s, X509Certificate c,
                                                          X509Chain chain, SslPolicyErrors e) {
            errors = e;
            if (chain != null) {
                foreach (X509ChainStatus st in chain.ChainStatus) {
                    chainStatus += (chainStatus.Length > 0 ? "," : "") + st.Status;
                }
            }
            // Always accept: the verdict comes from the policy errors.
            return true;
        };
        try {
            using (TcpClient client = new TcpClient()) {
                if (!client.ConnectAsync(host, 443).Wait(30000)) {
                    return "INCONCLUSIVE|connect timed out";
                }
                using (SslStream ssl = new SslStream(client.GetStream(), false, cb)) {
                    ssl.AuthenticateAsClient(host);
                }
            }
        } catch (Exception ex) {
            return "INCONCLUSIVE|" + ex.Message;
        }
        if (errors == SslPolicyErrors.None) { return "FAIL|"; }
        return "PASS|" + errors + "; chain: " + chainStatus;
    }
}
"@

function Test-HostTrust([string]$HostName) {
    $parts = [EmudhraTrustProbe]::Probe($HostName).Split("|", 2)
    return @{ Result = $parts[0]; Detail = $parts[1] }
}

Write-Output "--- Probing for trust of the compromised eMudhra intermediates"
$status = 0
foreach ($h in $Hosts) {
    $r = Test-HostTrust $h
    switch ($r.Result) {
        "PASS" {
            Write-Output "PASS: $h rejected (compromised CA is not trusted) [$($r.Detail)]"
        }
        "FAIL" {
            Write-Output "FAIL: $h ACCEPTED - this environment still trusts the compromised CAs"
            $status = 1
        }
        default {
            Write-Output "INCONCLUSIVE: $h (could not complete the check): $($r.Detail)"
            if ($status -eq 0) { $status = 2 }
        }
    }
}
exit $status
