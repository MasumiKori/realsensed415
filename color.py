import numpy as np
import pyrealsense2 as rs
import cv2

config = rs.config()
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)

pipeline = rs.pipeline()
profile = pipeline.start(config)

try:
    
        # フレーム待ち(Color & Depth)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
       
           
        color_image = np.asanyarray(color_frame.get_data())
        
        image = color_image
        cv2.namedWindow('RealSense',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense',image)
        if cv2.waitKey(1) & 0xff == 27:
        	break
        	
finally:
	pipeline.stop()
	cv2.destroyAllWindows()
        