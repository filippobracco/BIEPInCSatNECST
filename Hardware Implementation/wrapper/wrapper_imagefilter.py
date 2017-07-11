import numpy as np
from pynq import Overlay
ol = Overlay("biepincs.bit")
ol.download()
ol.bitstream.timestamp
from pynq.drivers import DMA
import numpy as np
from cffi import FFI
ffi = FFI()
W = 50
CH = 3
dmaOut = DMA(0x40400000,1)
dmaIn = DMA(0x40400000,0)
dmaIn.create_buf(W*W*CH)
dmaOut.create_buf(W*W)
pointIn = ffi.cast("uint8_t *", dmaIn.get_buf())
pointOut = ffi.cast("uint8_t *", dmaOut.get_buf())
c_buffer = ffi.buffer(pointOut, W*W)
def image_filter(th, image_in):
    pointIn[0] = th
    image = image_in.copy()
    pointerToCvimage = ffi.cast("uint8_t *", ffi.from_buffer(image))
    ffi.memmove(pointIn+1, pointerToCvimage, W*W*CH)
    dmaOut.transfer(W*W, 1)
    dmaIn.transfer(W*W*CH, 0)
    dmaOut.wait()
    result = np.frombuffer(c_buffer, count=W*W ,dtype=np.uint8)
    result = result.reshape(W,W)
    return result   

