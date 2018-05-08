

""" 
layer {
    name: "softmax"
    type: "Softmax"
    bottom: "fc1_data"
    top: "prob"
}


layer {
  type: 'Python'
  name: 'loss'
  top: 'loss'
  bottom: 'prob'
  #bottom: 'fc1_data'
  bottom: 'label'
  python_param {
    module: 'dice_loss' # the module name -- usually the filename -- that needs to be in $PYTHONPATH
    layer: 'DiceLoss' # the layer name -- the class name in the module
  }
  loss_weight: 1 # set loss weight so Caffe knows this is a loss layer
  # force_backward: true
}
"""

import caffe
import numpy as np
import warnings
import math

class DiceLoss(caffe.Layer):
    """
    This loss layer gives per pixel loss which means it will give better segmentation in cases where there is high imbalance between pixels
    of fore-ground class and back-ground class. Designed for a batch size of 1 at present but this can be further modified.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute dice loss.")


    def reshape(self, bottom, top):

        if bottom[0].count != 2*(bottom[1].count):
            raise Exception("The number of elements have to be twice the number of the elements of the ground truth.")

        self.gt = np.zeros_like(bottom[0].data[0][0],dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.gt[...]=bottom[1].data[0][0]
        self.pred = bottom[0].data[0][1]
        self.union_fg= self.pred.sum() + bottom[1].data.sum()
        self.intersection_fg = 2.* ((self.pred * bottom [1].data).sum())
        self.dice = self.intersection_fg / (self.union_fg + 0.00001)
        top[0].data[...] = self.dice


    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("label not diff")
        elif propagate_down[0]:

            bottom[0].diff[0,0,...] = 2.0 * ( 
                    (self.gt * self.union_fg) - (self.intersection_fg))/(
                    ((self.union_fg) ** 2) + 0.0001)
            bottom[0].diff[0,1,...] = -2.0 * ( 
                    (self.gt * self.union_fg) - (self.intersection_fg))/(
                    ((self.union_fg) ** 2) + 0.0001)



        else:
            raise Exception("no diff")

