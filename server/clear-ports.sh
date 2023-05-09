#!/usr/bin/env bash
for port in {6000..6012}; do
    pids=$(sudo lsof -t -i:$port)
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            user=$(ps -p $pid -o user=)
            sudo kill $pid
            echo "Killed process $pid on port $port started by user $user"
        done
    fi
done
