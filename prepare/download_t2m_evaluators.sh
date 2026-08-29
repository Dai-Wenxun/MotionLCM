mkdir -p deps/
cd deps/

echo "The t2m evaluators will be stored in the './deps' folder"

echo "Downloading"
gdown 16hyR4XlEyksVyNVjhIWK684Lrm_7_pvX -O t2m.zip
echo "Extracting"
unzip t2m.zip
echo "Cleaning"
rm t2m.zip

echo "Downloading done!"
