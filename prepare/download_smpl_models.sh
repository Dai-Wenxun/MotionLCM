mkdir -p deps/
cd deps/

echo -e "Downloading smpl models"
gdown --fuzzy https://drive.google.com/file/d/1J2pTxrar_q689Du5r3jES343fZUmCs_y/view?usp=sharing
rm -rf smpl_models

unzip smpl_models.zip
echo -e "Cleaning\n"
rm smpl_models.zip

echo -e "Downloading done!"