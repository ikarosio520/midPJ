import numpy as np
import serial
import time

waitTime = 0.1

signalLength = 84
#Song database
song0 = np.array([261, 261, 392, 392, 440, 440, 392, 349, 349, 330, 330, 294, 294, 261, 392, 392, 349, 349, 330, 330, 294, 392, 392, 349, 349, 330, 330, 294, 261, 261, 392, 392, 440, 440, 392, 349, 349, 330, 330, 294, 294, 261])
noteLength0 = np.array([1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2])
song1 = np.array([261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 330, 330, 294, 392, 392, 349, 349, 330, 330, 294, 261, 261, 392, 392, 440, 440, 392, 349, 349, 330, 330, 294, 294, 261])
noteLength1 = np.array([1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2])
song2 = np.array([261, 261, 392, 392, 440, 440, 392, 349, 349, 330, 330, 294, 294, 261, 392, 392, 349, 349, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261])
noteLength2 = np.array([1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2])

# output formatter
formatter = lambda x: "%d" % x
# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("get song_index signal ...")
line=s.readline() # Read an echo string from K66F terminated with '\n'
song_index = int(line)
print("get! index %d" % (int(song_index)))
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
if song_index == 0:
  for data in song0:
    s.write(bytes(formatter(data), 'UTF-8'))
    #print(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)
  for data in noteLength0:
    s.write(bytes(formatter(data), 'UTF-8'))
    #print(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)
if song_index == 1:
  for data in song1:
    s.write(bytes(formatter(data), 'UTF-8'))
    #print(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)
  for data in noteLength1:
    s.write(bytes(formatter(data), 'UTF-8'))
    #print(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)
if song_index == 2:
  for data in song2:
    s.write(bytes(formatter(data), 'UTF-8'))
    #print(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)
  for data in noteLength2:
    s.write(bytes(formatter(data), 'UTF-8'))
    #print(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)
s.close()
print("Signal sended")