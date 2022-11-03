One of the important goals of the project, is to have as much of the same codebase running on both the device at inference time, and the training host box at train/debug time.

We accomplish this partly by exporting all complex calculations to ONNX files, and verifying/running that they match on both the host and device.
Another big aspect of this is converting a stream of messages into a vector to be passed into an RL model.

*Runtime*
 - Each vision frame is run through the vision intermediate network
 - Recieving vision frames trigger evaluation of the RL model
 - All of the received messages and videos feed into the current MsgVec instance, which outputs a single observation tensor
 - Log a "I ran the RL model" type message with timestamps
 - Observation tensor gets run through the ML model and an action vector is produced
 - Action vectors are converted out cereal messages and send to actual controls
 

*Server*
 - Go through a huge batch of log files, and for each model inference message, recreate exactly the same observation tensors that went into the RL models
 - Also, recreate the action vector
  - You'd be tempted to just log it and read it back, but you'd want this to work even if you had a non RL model controlling the robot
 - Build the full [obs, act, reward, done] matrix and cache that for easy loading of subsequent training runs


*Things we need to support eventually*
 - Audio (convert last N messages to FFT coeffs, maybe run through a network, and pass that to the RL)
 - Sending action vectors as deltas to the previous value, instead of absolute commands
  (ex. send turn a little more to the left, rather than setting a new position value for the camera)
 - Adding timing information as an extra obs field
    - Ex. Was that last motor feedback message from 2 or 20ms ago?
