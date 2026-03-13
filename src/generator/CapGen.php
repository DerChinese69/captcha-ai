<?php
// Captcha generator using Gregwar Captcha library
// Generates captcha images and writes labels to a CSV file.
// To run: php data/CapGen.php [char_length] [num_samples] [width] [height] [charset]
// Data will be saved in /data/raw/ with folder naming convention: 4Char_100000_CapGen

require_once __DIR__ . '/../../vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;
use Gregwar\Captcha\PhraseBuilder;

/*
|--------------------------------------------------------------------------
| DEFAULT CONFIGURATION
|--------------------------------------------------------------------------
*/

$defaultLength = 4;
$defaultNumSamples = 10;
$defaultCharset = '1234679ACDEFGHJKMNPQRTUVWXYZabcdefghjkmnpqrtuvwxyz';
$defaultWidth = 192;
$defaultHeight = 64;


/*
|--------------------------------------------------------------------------
| COMMAND LINE ARGUMENTS
|--------------------------------------------------------------------------
*/

$length = isset($argv[1]) ? (int)$argv[1] : $defaultLength;
$numSamples = isset($argv[2]) ? (int)$argv[2] : $defaultNumSamples;
$width = isset($argv[3]) ? (int)$argv[3] : $defaultWidth;
$height = isset($argv[4]) ? (int)$argv[4] : $defaultHeight;
$charset = isset($argv[5]) ? $argv[5] : $defaultCharset;



/*
|--------------------------------------------------------------------------
| OUTPUT DIRECTORY
|--------------------------------------------------------------------------
*/

$rootDir = __DIR__ . '/../../data/raw/';
$runFolder = $length . "Char_" . $numSamples . "_CapGen";
$outputDir = $rootDir . $runFolder . '/';

if (!file_exists($outputDir)) {
    mkdir($outputDir, 0777, true);
}


/*
|--------------------------------------------------------------------------
| CREATE CSV FILE FOR LABELS
|--------------------------------------------------------------------------
| The CSV will be placed at the top of the run folder.
*/

$csvPath = $outputDir . 'ground_truth_index.csv';
$csvFile = fopen($csvPath, 'w');

// Write CSV header
fputcsv($csvFile, ['filename', 'label']);


/*
|--------------------------------------------------------------------------
| PHRASE BUILDER
|--------------------------------------------------------------------------
*/

$phraseBuilder = new PhraseBuilder($length, $charset);


/*
|--------------------------------------------------------------------------
| GENERATION START
|--------------------------------------------------------------------------
*/

echo "Starting generation...\n";
echo "Length: {$length}\n";
echo "Samples: {$numSamples}\n";
echo "Charset: {$charset}\n";
echo "Output directory: {$outputDir}\n\n";


/*
|--------------------------------------------------------------------------
| CAPTCHA GENERATION LOOP
|--------------------------------------------------------------------------
*/
echo "\nWriting CSV file...\n";

for ($i = 0; $i < $numSamples; $i++) {

    $builder = new CaptchaBuilder(null, $phraseBuilder);
    $builder->build($width, $height);

    $phrase = $builder->getPhrase();

    $filename = str_pad($i + 1, 6, '0', STR_PAD_LEFT) . '.png';
    $filepath = $outputDir . $filename;

    // Save image
    $builder->save($filepath);

    // Write label entry to CSV
    fputcsv($csvFile, [$filename, $phrase], ',', '"', '');

    // echo '[' . ($i + 1) . '/' . $numSamples . "] $filename\n";
    if ((($i + 1) % 1000) === 0 || ($i + 1) === $numSamples) {
    echo '[' . ($i + 1) . '/' . $numSamples . "]\n";
    }
}


/*
|--------------------------------------------------------------------------
| CLOSE CSV FILE
|--------------------------------------------------------------------------
*/

fclose($csvFile);
echo "\nCSV file written.\n";
echo "\nGeneration complete.\n";
echo "Labels saved to: {$csvPath}\n";