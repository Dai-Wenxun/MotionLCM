echo -e "Downloading experiments_t2m!"
# experiments_t2m
gdown --fuzzy https://drive.google.com/file/d/1RpNumy8g2X2lI4H-kO9xkZVYRDDgyu6Y/view?usp=sharing
unzip experiments_t2m.zip

echo -e "Downloading experiments_control!"
# experiments_control
gdown --fuzzy https://drive.google.com/file/d/1_xrSGWHo4pRz-AJj22rt8qmRX0SflJ6V/view?usp=sharing
unzip experiments_control.zip

rm experiments_t2m.zip
rm experiments_control.zip

echo -e "Downloading done!"
