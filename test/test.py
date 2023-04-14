import os
import re

import cv2

templates_dir = r'D:\fy.xie\fenx\fenx - General\Ubei\Stereo\init_palte\template'
template_images = [(int(re.findall('\d+', f)[0]), cv2.imread(os.path.join(templates_dir, f), cv2.IMREAD_GRAYSCALE)) for f in os.listdir(templates_dir) if 'template' in f]
print(template_images)
