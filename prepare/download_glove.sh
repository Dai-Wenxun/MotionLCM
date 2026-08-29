mkdir -p deps/
cd deps/

echo -e "Downloading glove (in use by the evaluators)"
gdown 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n -O glove.zip
rm -rf glove

unzip glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"