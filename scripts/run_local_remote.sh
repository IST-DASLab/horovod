#!/usr/bin/env bash

hosts="dasgpu2 dasgpu3 dasgpu4"
#bash -c "$1"
for host in $hosts; do
    ssh $host hostname
    ssh $host $1
done