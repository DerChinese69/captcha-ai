#!/bin/zsh

NUM_SAMPLES=10

for LENGTH in 5 6 7 8 9 10
do
  echo "Starting ${LENGTH}-char generation..."
  php ./src/generator/CapGen.php $LENGTH $NUM_SAMPLES
  echo "Finished ${LENGTH}-char generation."
done

echo "All generations complete."