#!/usr/bin/env bash

curl -i --header "Content-Type: application/json" --request POST \
--data '{"application":"data/raw/application_test.csv.zip","bureau":"data/raw/bureau.csv.zip"}' \
http://localhost:5000/predict


