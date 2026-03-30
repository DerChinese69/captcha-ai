# CapGen.php

Captcha generator using Gregwar Captcha library
Generates captcha images and writes labels to a CSV file.
To run: php data/CapGen.php [char_length] [num_samples] [width] [height] [charset]
Data will be saved in /data/raw/ with folder naming convention: 4Char_100000_CapGen

# run_all_generations
**Functioning**
Runs a zsh/bash loop for CapGen.php character lengths selected. 
Variables have to be changed in the script.

**Commands to Run in Terminal**
*Run with caffeinate + background logging:*
nohup /usr/bin/caffeinate /bin/zsh ./src/generator/run_all_generations.sh > generation_log.txt 2>&1 &

*Watch Progress:*
tail -f generation_log.txt

*One line force kill* (Was the only effective method for me)
ps aux | grep -E "run_all_generations|generate_captchas|caffeinate" | grep -v grep | awk '{print $2}' | xargs kill -9


