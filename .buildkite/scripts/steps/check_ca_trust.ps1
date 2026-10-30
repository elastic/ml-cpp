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
# Also checks the JDK's own cacerts (what Gradle uses) and the CA bundles that
# other tools install (Git, curl, Python); Schannel does not cover either.
# Exit status: 0 = all pass, 1 = any FAIL, 2 = inconclusive only, 3 = only one of
# those other stores trusts them (soft-failed until the image is rebuilt).

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
        // A hostname mismatch alone says nothing about the chain.
        if (errors == SslPolicyErrors.RemoteCertificateNameMismatch) {
            return "INCONCLUSIVE|hostname mismatch only";
        }
        return "PASS|" + errors + "; chain: " + chainStatus;
    }
}
"@

function Test-HostTrust([string]$HostName) {
    $parts = [EmudhraTrustProbe]::Probe($HostName).Split("|", 2)
    return @{ Result = $parts[0]; Detail = $parts[1] }
}

# SHA-256 of the two roots, as keytool prints them.
$BadFingerprints = @(
    "D7:A7:A0:FB:5D:7E:27:31:D7:71:E9:48:4E:BC:DE:F7:1D:5F:0C:3E:0A:29:48:78:2B:C8:3E:E0:EA:69:9E:F4",
    "86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B"
)

# The JDK Gradle would use (JAVA_HOME, else PATH): "PASS", "FAIL" or "INCONCLUSIVE".
function Test-JavaTrust {
    $keytool = $null
    if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME "bin\keytool.exe"))) {
        $keytool = Join-Path $env:JAVA_HOME "bin\keytool.exe"
    } else {
        $cmd = Get-Command keytool -ErrorAction SilentlyContinue
        if ($cmd) { $keytool = $cmd.Source }
    }
    if (-not $keytool) {
        return @{ Result = "INCONCLUSIVE"; Detail = "keytool not found (JAVA_HOME unset, not on PATH)" }
    }
    $ErrorActionPreference = "Continue"   # keytool writes warnings to stderr
    $out = & $keytool -list -v -cacerts -storepass changeit 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) {
        return @{ Result = "INCONCLUSIVE"; Detail = "keytool failed: $($out | Select-Object -First 1)" }
    }
    $fps = @($out | ForEach-Object { if ($_ -match "SHA256:\s*([0-9A-F:]+)") { $Matches[1] } })
    if ($fps.Count -lt 50) {
        return @{ Result = "INCONCLUSIVE"; Detail = "only $($fps.Count) certificates listed by $keytool" }
    }
    $found = @($fps | Where-Object { $BadFingerprints -contains $_ })
    if ($found.Count -gt 0) {
        return @{ Result = "FAIL"; Detail = "$keytool lists $($found.Count) of the roots" }
    }
    return @{ Result = "PASS"; Detail = "$keytool, $($fps.Count) certificates" }
}

Write-Output "--- Probing the JDK's cacerts"
$j = Test-JavaTrust
$javaFailed = $false
switch ($j.Result) {
    "PASS" { Write-Output "PASS: Java cacerts does not trust the compromised roots [$($j.Detail)]" }
    "FAIL" {
        Write-Output "FAIL: Java cacerts TRUSTS the compromised roots [$($j.Detail)] (known gap)"
        $javaFailed = $true
    }
    default { Write-Output "INCONCLUSIVE: Java cacerts ($($j.Detail))" }
}

# PEM CA bundles installed by other tools (Git for Windows, curl, Python's certifi):
# returns one object per bundle of 50+ certificates, with how many are the roots.
function Test-CaBundles([string[]]$Roots, [string[]]$Names) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $files = @()
    foreach ($root in $Roots) {
        if (Test-Path $root) {
            $files += Get-ChildItem -Path $root -Recurse -Include $Names -File -ErrorAction SilentlyContinue
        }
    }
    foreach ($f in ($files | Sort-Object FullName -Unique)) {
        $text = Get-Content -Raw -ErrorAction SilentlyContinue $f.FullName
        if (-not $text) { continue }
        $certs = 0
        $bad = 0
        foreach ($m in [regex]::Matches($text, "-----BEGIN CERTIFICATE-----([^-]+)-----END CERTIFICATE-----")) {
            try { $der = [Convert]::FromBase64String(($m.Groups[1].Value -replace "\s", "")) } catch { continue }
            $certs++
            $fp = ($sha.ComputeHash($der) | ForEach-Object { $_.ToString("X2") }) -join ":"
            if ($BadFingerprints -contains $fp) { $bad++ }
        }
        if ($certs -ge 50) { [pscustomobject]@{ Path = $f.FullName; Certs = $certs; Bad = $bad } }
    }
}

$BundleRoots = @(
    "$env:ProgramFiles\Git", "${env:ProgramFiles(x86)}\Git", "$env:ProgramData\chocolatey",
    "$env:ProgramFiles\Python*", "C:\Python*", "$env:LOCALAPPDATA\Programs\Python", "C:\msys64", "C:\tools"
)
$BundleNames = @("cacert.pem", "curl-ca-bundle.crt", "ca-bundle.crt", "ca-certificates.crt", "cert.pem")

Write-Output "--- Probing CA bundles installed by other tools"
$bundles = @(Test-CaBundles $BundleRoots $BundleNames)
$bundlesFailed = $false
foreach ($b in $bundles) {
    if ($b.Bad -gt 0) {
        Write-Output "FAIL: $($b.Path) contains $($b.Bad) of the compromised roots ($($b.Certs) certificates)"
        $bundlesFailed = $true
    } else {
        Write-Output "PASS: $($b.Path) ($($b.Certs) certificates)"
    }
}
if ($bundles.Count -eq 0) { Write-Output "No CA bundle files found under the scanned directories" }

Write-Output "--- Probing for trust of the compromised eMudhra intermediates"

# The control host must be accepted (Probe reports that as "FAIL"), otherwise
# validation is not working here and a rejection proves nothing.
$ControlHost = "github.com"
$c = Test-HostTrust $ControlHost
if ($c.Result -ne "FAIL") {
    Write-Output "INCONCLUSIVE: control host $ControlHost was not accepted, so this environment cannot vouch for the result [$($c.Detail)]"
    exit $(if ($javaFailed -or $bundlesFailed) { 3 } else { 2 })
}

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
if (($javaFailed -or $bundlesFailed) -and $status -ne 1) { $status = 3 }
exit $status
