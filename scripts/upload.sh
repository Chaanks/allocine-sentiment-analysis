#!/bin/bash

curl -X PUT "localhost:9200/movie_db" -H 'Content-Type: application/json' -d "@$1"

for file in $2/*.ndjson; do
    echo "upload $file"
    curl -o /dev/null -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/movie_db/_bulk?pretty' --data-binary @$file
done
