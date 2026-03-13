#!/bin/zsh

NUM_SAMPLES=100000

for LENGTH in 4 5 6 7 8 9 10
do
  echo "Starting ${LENGTH}-char generation..."
  START_TIME=$(date +%s)

  php ./src/generator/CapGen.php $LENGTH $NUM_SAMPLES

  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))

  HOURS=$((ELAPSED / 3600))
  MINUTES=$(((ELAPSED % 3600) / 60))
  SECONDS=$((ELAPSED % 60))

  echo "Finished ${LENGTH}-char generation."
  echo "Time spent: ${HOURS}h ${MINUTES}m ${SECONDS}s"
done

echo "All generations complete."