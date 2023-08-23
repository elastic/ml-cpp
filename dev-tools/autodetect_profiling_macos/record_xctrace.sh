#!/bin/bash

while true; do
  autodetect_pid=$(pgrep autodetect)

  if [[ -n $autodetect_pid ]]; then
    echo "Process 'autodetect' found with PID $autodetect_pid."

    # Start 'xctrace' to record memgraph
    xctrace_command="xctrace record --output autodetect_${autodetect_pid}.trace --template 'Allocations' --attach 'autodetect'"
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
