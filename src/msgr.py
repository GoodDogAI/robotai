import cereal.messaging as messaging

pm = messaging.PubMaster(['sensorEvents'])
sm = messaging.SubMaster(['sensorEvents'])

dat = messaging.new_message('sensorEvents', size=1)
dat.sensorEvents[0] = {"gyro": {"v": [1.1, -0.1, 0.1]}}
pm.send('sensorEvents', dat)


# in subscriber

while 1:
  sm.update()
  print(sm['sensorEvents'])
  print(sm.updated)