import time
import cereal.messaging as messaging

pm = messaging.PubMaster(['sensorEvents'])

x = 1

while 1:
    dat = messaging.new_message('sensorEvents', size=1)
    dat.sensorEvents[0] = {"gyro": {"v": [x + 0.1, -0.1, 0.1]}}
    pm.send('sensorEvents', dat)
    print("Sent message")
    x += 1
    time.sleep(3)