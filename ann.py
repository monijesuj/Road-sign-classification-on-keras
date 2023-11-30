#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
from keras.models import model_from_json


def main():
    # Parse command line arguments
    arg_parser = argparse.ArgumentParser(description='keras model test')
    arg_parser.add_argument(
        '--model-file',
        required=True,
        help='model json file',
    )
    arg_parser.add_argument(
        '--weights-file',
        required=True,
        help='model weights file',
    )
    arg_parser.add_argument(
        '--video-type',
        choices=['file', 'camera'],
        default='camera',
        help='video type',
    )
    arg_parser.add_argument(
        '--source',
        default='/dev/video0',
        help='source file or camera device',
    )
    arg_parser.add_argument(
        '--input-width',
        type=int,
        default=30,
        help='Input image width',
    )
    arg_parser.add_argument(
        '--input-height',
        type=int,
        default=30,
        help='Input image height',
    )
    arg_parser.add_argument(
        '--gui',
        action='store_true',
        help='Show GUI',
    )
    args = arg_parser.parse_args()
    assert args.input_width > 0 and args.input_height > 0

 
    with open(args.model_file, 'r') as file_model:
        model_desc = file_model.read()
        model = model_from_json(model_desc)

    model.load_weights(args.weights_file)

  
    if args.video_type == 'file': 
        video_dev = cv2.VideoCapture(args.source)
        video_width = video_dev.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT)

    elif args.video_type == 'camera':  
        video_dev = cv2.VideoCapture(0)

    try:
        prev_timestamp = time.time()

        while True:
            ret, orig_image = video_dev.read()
            curr_time = time.localtime()

            
            if ret is None or orig_image is None:
                break
            
            resized_image = cv2.resize(
                orig_image,
                (args.input_width, args.input_height),
            ).astype(np.float32)
            normalized_image = resized_image / 255.0

       
            batch = normalized_image.reshape(1, args.input_height, args.input_width, 3)
            result_onehot = model.predict(batch)
            speed2, speed3, speed5, speed6, speed7, speed8, speed_8, speed10, speed12, no_passing, no_passing_3, rotw,\
                priority_road, Yield, stop, no_vehicles, nv_35, no_entry, general_caution, dc_right,\
                    dc_left, double_curve, bumpy_road, slippery_road, rnotright, road_work,\
                        traffic_signals, pedestrians, children_cross, bicycles, ice_snow,\
                            wild, endOfAll, right_ahead, left_ahead, ahead_only,\
                                straigt_right, straigt_left, keepr, keepl, roundabout,\
                                    endonp, endonp3 = result_onehot[0]
            class_id = np.argmax(result_onehot, axis=1)[0]
            classNo = class_id
            if   classNo == 0:
                 class_str =  'Speed Limit 20 km/h'
            elif classNo == 1: 
                class_str = 'Speed Limit 30 km/h'
            elif classNo == 2: 
                class_str =  'Speed Limit 50 km/h'
            elif classNo == 3: 
                class_str = 'Speed Limit 60 km/h'
            elif classNo == 4: 
                class_str == 'Speed Limit 70 km/h'
            elif classNo == 5: 
                class_str =  'Speed Limit 80 km/h'
            elif classNo == 6: 
                class_str = 'End of Speed Limit 80 km/h'
            elif classNo == 7: 
                class_str = 'Speed Limit 100 km/h'
            elif classNo == 8: 
                class_str = 'Speed Limit 120 km/h'
            elif classNo == 9: 
                class_str = 'No passing'
            elif classNo == 10: 
                class_str =  'No passing for vechiles over 3.5 metric tons'
            elif classNo == 11: 
                class_str = 'Right-of-way at the next intersection'
            elif classNo == 12:
                class_str = 'Priority road'
            elif classNo == 13:
                class_str = 'Yield'
            elif classNo == 14:
                class_str = 'Stop'
            elif classNo == 15:
                class_str = 'No vechiles'
            elif classNo == 16: 
                class_str = 'Vechiles over 3.5 metric tons prohibited'
            elif classNo == 17:
                class_str = 'No entry'
            elif classNo == 18:
                class_str = 'General caution'
            elif classNo == 19:
                class_str = 'Dangerous curve to the left'
            elif classNo == 20:
                class_str = 'Dangerous curve to the right'
            elif classNo == 21:
                class_str = 'Double curve'
            elif classNo == 22:
                class_str = 'Bumpy road'
            elif classNo == 23:
                class_str = 'Slippery road'
            elif classNo == 24:
                class_str = 'Road narrows on the right'
            elif classNo == 25:
                class_str = 'Road work'
            elif classNo == 26:
                class_str = 'Traffic signals'
            elif classNo == 27:
                class_str = 'Pedestrians'
            elif classNo == 28:
                class_str = 'Children crossing'
            elif classNo == 29:
                class_str = 'Bicycles crossing'
            elif classNo == 30:
                class_str = 'Beware of ice/snow'
            elif classNo == 31:
                class_str = 'Wild animals crossing'
            elif classNo == 32:
                class_str = 'End of all speed and passing limits'
            elif classNo == 33:
                class_str = 'Turn right ahead'
            elif classNo == 34:
                class_str = 'Turn left ahead'
            elif classNo == 35:
                class_str = 'Ahead only'
            elif classNo == 36:
                class_str = 'Go straight or right'
            elif classNo == 37:
                class_str = 'Go straight or left'
            elif classNo == 38:
                class_str = 'Keep right'
            elif classNo == 39:
                class_str = 'Keep left'
            elif classNo == 40:
                class_str = 'Roundabout mandatory'
            elif classNo == 41: 
                class_str = 'End of no passing'
            elif classNo == 42:
                class_str = 'End of no passing by vechiles over 3.5 metric tons'
            else:
                class_str = 'No sign'

            recent_timestamp = time.time()
            period = recent_timestamp - prev_timestamp
            prev_timestamp = recent_timestamp
            
            print('time:%02d:%02d:%02d ' % (curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec))
            print('fl:%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\
                 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % (\
                speed2,speed3, speed5, speed6, speed7, speed8, speed_8, speed10, speed12, no_passing, no_passing_3, rotw,\
                priority_road, Yield, stop, no_vehicles, nv_35, no_entry, general_caution, dc_right,\
                    dc_left, double_curve, bumpy_road, slippery_road, rnotright, road_work,\
                        traffic_signals, pedestrians, children_cross, bicycles, ice_snow,\
                            wild, endOfAll, right_ahead, left_ahead, ahead_only,\
                                straigt_right, straigt_left, keepr, keepl, roundabout,\
                                    endonp, endonp3))
            print('name:%s' % class_str)
            print('time:%f' % period)
            print()
            
            # display
            cv2.putText(orig_image, class_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('image', orig_image)
            cv2.waitKey(1) & 0xFF
            '''
            if args.gui:
                cv2.imshow('', orig_image)
                cv2.waitKey(1) & 0xFF
            '''
    except KeyboardInterrupt:
        print('terminated by user')

    # CLOSE
    video_dev.release()


if __name__ == '__main__':
    main()