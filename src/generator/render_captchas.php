<?php
require_once __DIR__ . '/../../vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;

/*
|--------------------------------------------------------------------------
| USAGE
|--------------------------------------------------------------------------
| php src/generator/render_captchas.php data/orders/order_0001 5Char_100000_CapGen_balanced
|
| Arg 1: order folder path
| Arg 2: output folder name inside data/raw/
*/

if ($argc < 3) {
    echo "Usage: php src/generator/render_captchas.php <order_folder> <output_folder_name>\n";
    exit(1);
}

// Progress logging interval
$logInterval = 1000;
$startTime = microtime(true);

$orderFolder = rtrim($argv[1], '/');
$outputFolderName = $argv[2];

$csvPath = $orderFolder . '/captcha_order.csv';
$configPath = $orderFolder . '/order_config.json';

if (!file_exists($csvPath)) {
    echo "Error: captcha_order.csv not found at {$csvPath}\n";
    exit(1);
}

if (!file_exists($configPath)) {
    echo "Error: order_config.json not found at {$configPath}\n";
    exit(1);
}

$config = json_decode(file_get_contents($configPath), true);

if ($config === null) {
    echo "Error: could not parse {$configPath}\n";
    exit(1);
}

$length = $config['captcha']['length'] ?? 5;
$width = $config['captcha']['width'] ?? 192;
$height = $config['captcha']['height'] ?? 64;
$imageFormat = $config['output']['image_format'] ?? 'png';
$totalRows = $config['generation']['total_samples'] ?? 0;

$rootDir = __DIR__ . '/../../data/raw/';
$outputDir = $rootDir . $outputFolderName . '/';

if (!file_exists($outputDir) && !mkdir($outputDir, 0777, true)) {
    echo "Error: could not create output directory {$outputDir}\n";
    exit(1);
}

$groundTruthPath = $outputDir . 'ground_truth_index.csv';
$groundTruthFile = fopen($groundTruthPath, 'w');

if (!$groundTruthFile) {
    echo "Error: could not create {$groundTruthPath}\n";
    exit(1);
}

// CSV header
fputcsv($groundTruthFile, ['filename', 'label'], ',', '"', '');

$orderFile = fopen($csvPath, 'r');
if (!$orderFile) {
    echo "Error: could not open {$csvPath}\n";
    fclose($groundTruthFile);
    exit(1);
}

// Read header
$header = fgetcsv($orderFile, 0, ',', '"', '');
if ($header === false) {
    echo "Error: could not read header from {$csvPath}\n";
    fclose($orderFile);
    fclose($groundTruthFile);
    exit(1);
}

$headerMap = array_flip($header);

if (
    !isset($headerMap['sample_id']) ||
    !isset($headerMap['label'])
) {
    echo "Error: captcha_order.csv must contain columns: sample_id,label\n";
    fclose($orderFile);
    fclose($groundTruthFile);
    exit(1);
}

echo "Starting render...\n";
echo "Order folder: {$orderFolder}\n";
echo "Output folder: {$outputDir}\n";
echo "Length: {$length}\n";
echo "Width x Height: {$width} x {$height}\n";
echo "Expected total: {$totalRows}\n\n";

$count = 0;

while (($row = fgetcsv($orderFile, 0, ',', '"', '')) !== false) {
    $sampleId = $row[$headerMap['sample_id']];
    $label = $row[$headerMap['label']];

    $filename = $sampleId . '.' . $imageFormat;
    $filepath = $outputDir . $filename;

    $builder = new CaptchaBuilder($label);
    $builder->build($width, $height);
    $builder->save($filepath);

    fputcsv($groundTruthFile, [$filename, $label], ',', '"', '');

    $count++;

    if (($count % $logInterval) === 0 || ($totalRows > 0 && $count === $totalRows)) {
        $elapsed = microtime(true) - $startTime;
        $rate = $count / max($elapsed, 1e-9);

        if ($totalRows > 0) {
            $remaining = max(0, $totalRows - $count);
            $etaSeconds = $remaining / max($rate, 1e-9);
            $etaMin = (int) floor($etaSeconds / 60);
            $etaSec = (int) floor(fmod($etaSeconds, 60));

            echo "{$count} rendered... ETA {$etaMin}m {$etaSec}s\n";
        } else {
            echo "{$count} rendered...\n";
        }
    }
}

fclose($orderFile);
fclose($groundTruthFile);

$totalElapsed = microtime(true) - $startTime;
$totalMin = (int) floor($totalElapsed / 60);
$totalSec = (int) floor(fmod($totalElapsed, 60));

echo "\nRender complete.\n";
echo "Images saved to: {$outputDir}\n";
echo "Labels saved to: {$groundTruthPath}\n";
echo "Total rendered: {$count}\n";
echo "Total runtime: {$totalMin}m {$totalSec}s\n";