#!/bin/bash

line=0
data_dir=/Volumes/MLData2/text2speech
cat books/*.txt | while read x; do
  echo "$x" | say -o $data_dir/outputs/line_${line}.aiff
  echo "$x" >$data_dir/inputs/line_${line}.txt
  line=$((line+1))
done
