#!/bin/bash
cd /server
# Wait 5 seconds to ensure the display is fully ready
sleep 5
# Loop through ports 4242 to 4252
for port in {4242..4245}; do
    java ch.idsia.scenarios.MainRun -ag ServerAgent:$port -server on &
    sleep 5
done
java ch.idsia.scenarios.MainRun -ag ServerAgent:4246 -server on