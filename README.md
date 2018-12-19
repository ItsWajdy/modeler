# modeler
A simple package to reconstruct a 3D model of an object from a video taken for that object

## Prerequisites
This package uses Python 3.x

Also, you'll need OpenCV and numpy which can be downloaded using the following commands
```
pip install numpy
pip install opencv-python
```

## Installing
To install modeler simply run this command from the command line:
```
pip install modeler
```
Or
```
pip install git+git://github.com/Wajdy759/modeler.git#egg=modeler
```
Then you can import it using:
```
import modeler
```

## Usage
modeler has a module called SfM which handles wraps everything you'll need
```
sfm = SfM(results_dir, video_already_converted, video_path, video_sampling_rate)
```
Where:
- `results_dir`: the directory where the module outputs the .ply file
- `video_already_converted`: a boolean set to true if the video used was already converted to images in 'input_images'
- `video_path`: in case video_already_converted is False, the module uses the video in 'video_path' as input
- `video_sampling_rate`: the frequency at which the module extracts images from the video to use in the actual reconstruction

- To actually run the algorithm and get the 3D model you'd use
```
sfm.find_structure_from_motion()
```

## Example
```
from modeler import SfM
sfm = SfM('results/', False, 'videos/vid1.mp4', 27)
sfm.structure_from_motion()
```
When the progrm terminates you'll get a set of .ply files in the folder named results
