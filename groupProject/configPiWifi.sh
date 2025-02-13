#!/bin/bash

# Check if SSID is provided as an argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <SSID> [password]"
    exit 1
fi

# Set SSID and password variables
SSID=$1
PASSWORD=$2

# Create the YAML configuration
if [ -z "$PASSWORD" ]; then
    cat <<EOL > /etc/netplan/50-cloud-init.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: true
      optional: true
  wifis:
    wlan0:
      dhcp4: yes
      optional: true
      access-points:
        "$SSID": {}
EOL
else
    cat <<EOL > /etc/netplan/50-cloud-init.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: true
      optional: true
  wifis:
    wlan0:
      dhcp4: yes
      optional: true
      access-points:
        "$SSID":
          password: "$PASSWORD"
EOL
fi

# Apply the new configuration
netplan apply
