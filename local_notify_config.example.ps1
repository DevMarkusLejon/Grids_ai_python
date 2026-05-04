# Copy this file to local_notify_config.ps1 and set your private phone push URL.
# local_notify_config.ps1 is ignored by Git so secrets/contact routes are not committed.
#
# Recommended simple option:
# 1. Install the ntfy app on your phone.
# 2. Subscribe to a long, private topic name.
# 3. Put that topic URL below, for example:
#
# $env:GRIDS_NOTIFY_URL = "https://ntfy.sh/grids-ai-your-private-random-topic"

$env:GRIDS_NOTIFY_URL = ""
