# SCN_3D_Feature_Extractor_

Here is something ro remember
From the BEV, y is downside and the height of the camera is 1.65m
I took -2 < y < 2 to crop the original data, maybe we can ignore the ground
I took  0 < z < 80  to crop the original data, the second vel lidar is ignored
I took  0 < x < 80  to crop the original data

<!--      z
          /\
          |     y(downside, into the surface)
          |  __
          |   /|
          |  /
          | /
          |/_________\
                     / x
-->
