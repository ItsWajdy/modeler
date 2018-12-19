from modeler import SfM


sfm = SfM('results/', True, 'videos/vid1.mp4', 27, debug_mode=False)
print('Done')
# sfm.find_structure_from_motion()
