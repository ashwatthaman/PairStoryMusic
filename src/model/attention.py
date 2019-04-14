import chainer
from chainer import Chain,Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
try:
    import cupy as xp
except ImportError:
    import numpy as xp
except ModuleNotFoundError:
    import numpy as xp

#@profile由来のエラー除去
try:
    profile
except NameError:
    def profile(func):
        def inner(*args,**kwargs):
            return func(*args,**kwargs)
        return inner

class Attention(Chain):
    def __init__(self,att_hidden):
        self.att_hidden = att_hidden#2*self.out_size
        super(Attention, self).__init__(
            #Att
            att_w=L.Linear(self.att_hidden, 1),
            # att_w=L.Linear(self.out_size,self.out_size),
        )


    def __call__(self,enc_h,dec_h):
        h, att_w = self.calcAtt(enc_h,dec_h)
        return h,att_w

    @profile
    def calcAtt(self,enc_h,dec_h):
        batch_size = int(dec_h.data.shape[0])

        max_length = max([len(e_h) for e_h in enc_h])
        dec_hx_arr = F.broadcast_to(
            F.reshape(
                dec_h,
                (batch_size, 1, self.att_hidden)
            ),
            (batch_size, max_length, self.att_hidden)
        )

        att_w = F.softmax(
            F.reshape(
                self.att_w(
                    F.reshape(
                        F.tanh(enc_h + dec_hx_arr),
                        (batch_size * max_length, self.att_hidden)
                    )
                ),
                (batch_size, max_length)
            )
        )
        h = F.reshape(
            F.batch_matmul(att_w, enc_h, transa=True),
            (batch_size, self.att_hidden)
        )
        return h, att_w

