#!/bin/bash
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