from cereal import log
import cereal.messaging as messaging

fname = "/media/card/alphalog-2022-5-29-22_29.log"
f = open(fname, "rb")

events = log.Event.read_multiple(f)

for evt in events:
    print(evt.which(), evt.headEncodeData.idx.encodeId)