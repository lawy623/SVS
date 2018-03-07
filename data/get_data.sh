# get KittiStereo 2015
wget http://kitti.is.tue.mpg.de/kitti/data_scene_flow.zip -P ./KittiStereo2015/
unzip KittiStereo2015/data_scene_flow.zip -d KittiStereo2015/
rm -f KittiStereo2015/data_scene_flow.zip

#get Kitti Raw Data
#The list is copied from https://github.com/mrharicot/monodepth .
wget -i KittiRaw/utils/KittiRawToDownload.txt -P ./KittiRaw/
unzip 'KittiRaw/*.zip' -d KittiRaw/
rm -f KittiRaw/*.zip


