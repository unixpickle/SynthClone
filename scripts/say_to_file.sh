#!/bin/bash

line=0
data_dir=/Volumes/MLData2/text2speech
cat books/*.txt | while read x; do
  if (( line % 1000 == 0 )); then
    echo "Processing line: $line"
  fi
  if [ -f $data_dir/inputs/line_${line}.txt ]; then
    line=$((line+1))
    continue
  fi
  echo "$x" | say -o $data_dir/outputs/line_${line}.aiff
  echo "$x" >$data_dir/inputs/line_${line}.txt
  line=$((line+1))
done
