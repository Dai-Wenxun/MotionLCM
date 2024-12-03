echo -e "Downloading experiments_recons!"
gdown --fuzzy https://drive.google.com/file/d/15zFDitcOLhjbQ0CaOoM-QNKQUeyJw-Om/view?usp=sharing
unzip experiments_recons.zip

echo -e "Downloading experiments_t2m!"
gdown --fuzzy https://drive.google.com/file/d/1U7homKobR2gaDLfL5flS3N0g7e0a_AQd/view?usp=sharing
unzip experiments_t2m.zip

echo -e "Downloading experiments_control!"
gdown --fuzzy https://drive.google.com/file/d/1o6oFdH5dgQJNB5J2rDGKCMw3FDr9gUkW/view?usp=sharing
unzip experiments_control.zip

rm experiments_recons.zip
rm experiments_t2m.zip
rm experiments_control.zip

echo -e "Downloading done!"
