mkdir -p deps/
cd deps/

echo -e "Downloading smpl models"
gdown 1J2pTxrar_q689Du5r3jES343fZUmCs_y -O smpl_models.zip
rm -rf smpl_models

unzip smpl_models.zip
echo -e "Cleaning\n"
rm smpl_models.zip

echo -e "Downloading done!"