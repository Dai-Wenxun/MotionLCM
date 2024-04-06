mkdir -p deps/
cd deps/

echo "The t2m evaluators will be stored in the './deps' folder"

echo "Downloading"
gdown --fuzzy https://drive.google.com/file/d/16hyR4XlEyksVyNVjhIWK684Lrm_7_pvX/view?usp=sharing
echo "Extracting"
unzip t2m.zip
echo "Cleaning"
rm t2m.zip

echo "Downloading done!"
