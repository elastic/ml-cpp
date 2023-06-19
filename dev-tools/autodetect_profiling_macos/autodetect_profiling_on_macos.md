# Anomaly detection profiling on MacOS <!-- omit in toc -->

---
- Author: Valeriy Khakhutskyy
- Date: 2023-06-15
---

- [Objective](#objective)
- [Recording Information](#recording-information)
  - [Using `leaks`](#using-leaks)
  - [Using `xctrace` with Allocations template](#using-xctrace-with-allocations-template)
- [Common issues](#common-issues)
  - [Changing the Executable Signature](#changing-the-executable-signature)
    - [If nothing works](#if-nothing-works)
    - [Sources](#sources)

## Objective

Profiling anomaly detection executables can be challenging because their execution depends on input from Java. This document describes how to do this using XCode Instruments.

To access Instruments, you must do the following:
```
XCode->Open Developer Tools->Instruments
```

## Recording Information

Instruments tools such as Allocations allow you to run an executable or attach to a process. However, trying to attach to a running process using the UI is cumbersome because the process may be short-lived and the UI doesn't support filtering by process name.

So we need a bash command that periodically checks if a process named `autodetect` exists. If so, it grabs the PID of that process. Then it immediately starts Xcode's recording utility to create a recording and save it to disk.

### Using `leaks`

```bash
#!/bin/bash
# filename: record_memgraph.sh
while true; do
  autodetect_pid=$(pgrep autodetect)
  
  if [[ -n $autodetect_pid ]]; then
    echo "Process 'autodetect' found with PID $autodetect_pid."
    
    # Start 'leaks' utility from Xcode to generate memgraph
    leaks_command="leaks $autodetect_pid --outputGraph=autodetect_${autodetect_pid}"
    eval $leaks_command
    
    if [[ $? -eq 0 ]]; then
      echo "Memgraph generated successfully."
      break
    else
      echo "Failed to generate memgraph."
      break
    fi
  fi
  
  # Sleep for 0.1 seconds before checking again
  sleep 0.1
done
```

The recording can be opened in the UI with
```sh 
open autodetect_XXXX.memgraph
```

For more information:
 - [Using Xcode's visual debugger and Instruments' modules to prevent memory overuse](https://rderik.com/blog/using-xcode-s-visual-debugger-and-instruments-modules-to-prevent-memory-overuse/)


### Using `xctrace` with Allocations template

```bash
#!/bin/bash
# filename: record_xctrace.sh

while true; do
  autodetect_pid=$(pgrep autodetect)

  if [[ -n $autodetect_pid ]]; then
    echo "Process 'autodetect' found with PID $autodetect_pid."

    # Start 'xctrace' to record memgraph
    xctrace_command="xctrace record --output autodetect_${autodetect_pid}.trace --template 'Allocations' --attach $autodetect_pid"
    eval $xctrace_command

    if [[ $? -eq 0 ]]; then
      echo "Trace file generated successfully."
      break
    else
      echo "Failed to generate the trace file."
      break
    fi
  fi

  # Sleep for 0.1 seconds before checking again
  sleep 0.1
done

```

## Common issues

### Changing the Executable Signature

You may receive the following error message when you select the executable:
```
Required kernel recording resources are in use by another document
```

To fix the permission problems, you need to apply **in zsh** the following command to the executable (instead of `pytorch_inference`):
```zsh
codesign -s - -v -f --entitlements =(echo -n '<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "[https://www.apple.com/DTDs/PropertyList-1.0.dtd](https://www.apple.com/DTDs/PropertyList-1.0.dtd)"\>
<plist version="1.0">
    <dict>
        <key>com.apple.security.get-task-allow</key>
        <true/>
    </dict>
</plist>') build/distribution/platform/darwin-aarch64/controller.app/Contents/MacOS/pytorch_inference
```

If this doesn't work, create an `entitlements.xml' file in VSCode with the following content
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd"\>
<plist version="1.0">
    <dict>
        <key>com.apple.security.get-task-allow</key>
        <true/>
    </dict>
</plist>
```

and then codesign using the xml file:
```zsh
codesign -s - -v -f --entitlements ./build/distribution/platform/darwin-aarch64/controller.app/Contents/MacOS/autodetect
```

#### If nothing works

Restart your computer in recovery mode (start the laptop and hold down the power button). Then go to Utilities->Terminal and run
```sh
csrutil disable
```

Don't forget to re-enable the signature check by running 
```sh
csrutil enable
```
when you are done.


#### Sources
1. https://stackoverflow.com/questions/74262518/why-required-kernel-recording-resources-are-in-use-by-another-document-error-i
2. https://sourceware.org/gdb/wiki/PermissionsDarwin#Sign_and_entitle_the_gdb_binary
