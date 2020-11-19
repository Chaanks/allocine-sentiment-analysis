#!/bin/bash

for file in $1/*.ndjson; do
    echo "upload $file"
    curl -o /dev/null -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/movie_db/_bulk?pretty' --data-binary @$file
done
