# Download Stanford Dataset
wget http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip
wget http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip

# Unzip Stanford Dataset
unzip Stanford40_JPEGImages.zip -d Stanford40/
unzip Stanford40_ImageSplits.zip -d Stanford40/

# Download TV Human Interaction (TV-HI)
wget http://www.robots.ox.ac.uk/~alonso/data/tv_human_interactions_videos.tar.gz
wget http://www.robots.ox.ac.uk/~alonso/data/readme.txt

# Unzip TV Human Interaction (TV-HI)
mkdir TV-HI
tar -xvf  'tv_human_interactions_videos.tar.gz' -C TV-HI
mv readme.txt 'TV-HI/readme.txt'

# Prepare Data
cd ..
python prepare_stanford.py
python optical_flow.py