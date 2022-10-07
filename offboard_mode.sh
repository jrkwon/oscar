rostopic pub -r 20 /mavros/setpoint_raw/attitude mavros_msgs/AttitudeTarget  '{orientation:  {x: 0.0, y: 0.0, z: 0.0}, body_rate: {x: 0.0,y: 0.0,z: 0.0}, thrust: 0}'
rosservice call /mavros/set_mode "base_mode: 0
custom_mode: 'OFFBOARD'"
