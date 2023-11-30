# Road-sign-classification-on-keras

Working of System:
The 2 wheels of chassis connected with two motors. Motor driver IC l293d is used to control the motor, one motor driver IC can control only two motors .So the proposed system used one l293d that is enough to control the motors. So the input for motor driver IC is given by the Raspberry pi and the output pins of the motor IC are connected to the motor of the chassis.
For movement of the car in forward and backward direction system will rotate the wheels in equal speed whereas to move left or right system will slow down the one wheel as compare to the other one according to the turning points i.e. if system need to rotate the turn in left direction system have to slow down the left wheel and if there is need to turn in right direction system have to slow down the right wheel. In the proposed system the input to the motor driver IC from is given from the GPIO pins for driving the left motor and GPIO pins is used to drive the right motor from L293d.
The ultrasonic sensors measure the distance to an obstacle, so no matter the sign the camera is seeings, it won’t move if the threshold of the ultrasonic sensor is already triggered.
The camera gets the images, sends to the modules that detects signs, which is then sent to the module that recognizes or classifies signs, then based on the result, the robot is controlled.
