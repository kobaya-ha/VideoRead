import chainer
import chainer.functions as F
import chainer.links as L


class C3D(chainer.Chain):

    """Convolution3Dnet"""

    #insize = 227

    def __init__(self):
        super(C3D, self).__init__(
            conv1=L.ConvolutionND(3,3,10,ksize = (3,3,3), stride= (1,3,3), pad = 0),
            conv2=L.ConvolutionND(3,10,15,ksize=5, stride=(1,2,2), pad = 0),
            conv3=L.ConvolutionND(3,15,15,ksize=3, stride=2, pad = 0),
            fc4=L.Linear(33750, 4096),
            fc5=L.Linear(4096, 4096),
            fc6=L.Linear(4096, 102),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.fc4(h)
        h = self.fc5(h)
        h = self.fc6(h)
 
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
