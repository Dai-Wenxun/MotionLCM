echo -e "Downloading experiments_recons!"
gdown 15zFDitcOLhjbQ0CaOoM-QNKQUeyJw-Om -O experiments_recons.zip
unzip experiments_recons.zip

echo -e "Downloading experiments_t2m!"
gdown 1U7homKobR2gaDLfL5flS3N0g7e0a_AQd -O experiments_t2m.zip
unzip experiments_t2m.zip

echo -e "Downloading experiments_control!"
gdown 1o6oFdH5dgQJNB5J2rDGKCMw3FDr9gUkW -O experiments_control.zip
unzip experiments_control.zip

rm experiments_recons.zip
rm experiments_t2m.zip
rm experiments_control.zip

echo -e "Downloading done!"
